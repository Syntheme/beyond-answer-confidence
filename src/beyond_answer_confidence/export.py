"""Package per-item results for sharing, without any dataset text.

The export copies the scored rows (``*.jsonl``, gzipped), the result files
(``*.json``, with private ``_``-prefixed keys removed) and saved comparator
predictions (``*.npz``) of the chosen experiments, the nearest-distractor
choices (integer positions, no text) and writes ``MANIFEST.json`` with the
SHA-256 digest of every file, ``README.md`` and ``DATASETS.md``.

What is never exported:

- request caches: they hold the full prompts, i.e. dataset text under the
  datasets' own licences (some non-commercial);
- any row, result file or distractor file carrying a text field
  (:data:`TEXT_FIELDS`, checked at every depth) or a long string
  (:data:`MAX_STRING`): the whole export is refused;
- logs and anything outside the chosen experiments' output directories.

All JSON is written strictly: non-finite floats become ``null``.

Nothing is uploaded; publishing the archive is a separate, manual step.
"""

import gzip
import hashlib
import json
import logging
import shutil
import tarfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from beyond_answer_confidence import __version__
from beyond_answer_confidence.data import intents, loaders, multiple_choice
from beyond_answer_confidence.json_output import dumps

logger = logging.getLogger(__name__)

SUFFIXES = (".jsonl", ".json", ".npz")
"""File types that are exported; everything else is skipped."""

TEXT_FIELDS = frozenset(
    {
        "answer_text",
        "context",
        "customer_message",
        "facts",
        "hypothesis",
        "input",
        "instructions",
        "message",
        "original",
        "paragraph",
        "paragraphs",
        "passage",
        "premise",
        "prompt",
        "question",
        "questions",
        "request",
        "sentence",
        "sentences",
        "state",
        "text",
        "utterance",
    }
)
"""Field names that would carry dataset text. At the top level of a row or
result file their presence alone stops the export; at deeper levels they stop
it when their value holds any string (so a label that happens to be called
``text`` may still map to a probability)."""

MAX_STRING = 100
"""Longest string (value or key, at any depth) allowed in an exported row.
Ids, labels and option letters are far shorter; longer strings are treated as
possible dataset text."""

MANIFEST = "MANIFEST.json"
MANIFEST_SENTINEL = {"package": "beyond-answer-confidence"}
"""Written into every manifest; only a directory whose manifest carries it
is treated as a previous export that may be replaced."""


class UnsafeExportError(ValueError):
    """Raised when a file could leak dataset text."""


@dataclass(frozen=True)
class DatasetNote:
    """Provenance of one dataset.

    Attributes:
        name: Dataset name.
        source: Where it is downloaded from.
        revision: Pinned revision.
        licence: Licence as stated by its source.
    """

    name: str
    source: str
    revision: str
    licence: str


DATASETS: tuple[DatasetNote, ...] = (
    DatasetNote(
        "Banking77",
        "github.com/PolyAI-LDN/task-specific-datasets",
        intents.BANKING77_COMMIT,
        "CC BY 4.0",
    ),
    DatasetNote(
        "CLINC150", "hf:clinc/clinc_oos", intents.CLINC150_REVISION, "CC BY 3.0"
    ),
    DatasetNote(
        "PopQA",
        "hf:akariasai/PopQA",
        loaders.REVISIONS["akariasai/PopQA"],
        "none stated",
    ),
    DatasetNote(
        "Daily Oracle",
        "hf:agentic-learning-ai-lab/daily-oracle",
        loaders.REVISIONS["agentic-learning-ai-lab/daily-oracle"],
        "CC BY 4.0",
    ),
    DatasetNote(
        "HotpotQA",
        "hf:hotpotqa/hotpot_qa",
        loaders.REVISIONS["hotpotqa/hotpot_qa"],
        "CC BY-SA 4.0",
    ),
    DatasetNote(
        "Quizbowl (QANTA)",
        "hf:community-datasets/qanta",
        loaders.REVISIONS["community-datasets/qanta"],
        "unknown",
    ),
    DatasetNote(
        "SelfAware",
        "hf:ShuoZheLi/SelfAware (mirror of github.com/yinzhangyue/SelfAware)",
        loaders.REVISIONS["ShuoZheLi/SelfAware"],
        "Apache-2.0",
    ),
    DatasetNote(
        "SimpleQA Verified",
        "hf:google/simpleqa-verified",
        loaders.REVISIONS["google/simpleqa-verified"],
        "MIT",
    ),
    DatasetNote(
        "TruthfulQA",
        "hf:truthfulqa/truthful_qa + github.com/sylinrl/TruthfulQA",
        f"{loaders.REVISIONS['truthfulqa/truthful_qa']} / "
        f"{loaders.TRUTHFULQA_CSV_COMMIT}",
        "Apache-2.0",
    ),
    DatasetNote(
        "TriviaQA",
        "hf:mandarjoshi/trivia_qa",
        loaders.REVISIONS["mandarjoshi/trivia_qa"],
        "unknown",
    ),
    DatasetNote(
        "AmbigQA",
        "hf:sewon/ambig_qa",
        loaders.REVISIONS["sewon/ambig_qa"],
        "CC BY-SA 3.0",
    ),
    DatasetNote(
        "ChaosNLI",
        "hf:earino/chaosnli (mirror; see `beyond-answer-confidence check-mirrors`)",
        loaders.REVISIONS["earino/chaosnli"],
        "CC BY-NC 4.0 (non-commercial)",
    ),
    DatasetNote(
        "MMLU-Redux 2.0",
        "hf:edinburgh-dawg/mmlu-redux-2.0",
        multiple_choice.REVISIONS["edinburgh-dawg/mmlu-redux-2.0"],
        "CC BY 4.0",
    ),
    DatasetNote(
        "MMLU-CF",
        "hf:microsoft/MMLU-CF",
        multiple_choice.REVISIONS["microsoft/MMLU-CF"],
        "CDLA-Permissive-2.0",
    ),
    DatasetNote(
        "ANLI",
        "hf:facebook/anli",
        multiple_choice.REVISIONS["facebook/anli"],
        "CC BY-NC 4.0 (non-commercial)",
    ),
)
"""Datasets the experiments download, with pinned revisions and licences."""


def strip_private(obj: Any) -> Any:
    """Drop keys starting with ``_`` (internal arrays such as bootstrap draws).

    Args:
        obj: Parsed JSON.

    Returns:
        The object without private keys.
    """
    if isinstance(obj, dict):
        return {
            k: strip_private(v) for k, v in obj.items() if not str(k).startswith("_")
        }
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def long_string(obj: Any, limit: int = MAX_STRING) -> str | None:
    """Return the first string (value or key) longer than ``limit``, if any.

    Args:
        obj: Parsed JSON.
        limit: Longest allowed string.

    Returns:
        The offending string, or ``None``.
    """
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            if len(x) > limit:
                return x
        elif isinstance(x, dict):
            stack.extend(x.keys())
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    return None


def _holds_string(obj: Any) -> bool:
    """Return whether a value is, or contains, a string."""
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            return True
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    return False


def text_field(obj: Any) -> str | None:
    """Return the path of the first text field in a document, if any.

    A :data:`TEXT_FIELDS` key is reported at the top level whatever its
    value, and at any deeper level when its value holds a string.

    Args:
        obj: Parsed JSON (a row or a result document).

    Returns:
        A dotted path to the offending key, or ``None``.
    """
    stack: list[tuple[Any, str, int]] = [(obj, "", 0)]
    while stack:
        x, path, depth = stack.pop()
        if isinstance(x, dict):
            for k, v in x.items():
                where = f"{path}.{k}" if path else str(k)
                if k in TEXT_FIELDS and (depth == 0 or _holds_string(v)):
                    return where
                stack.append((v, where, depth + 1))
        elif isinstance(x, list):
            stack.extend((v, f"{path}[{i}]", depth + 1) for i, v in enumerate(x))
    return None


def check_document(obj: Any, where: str, limit: int = MAX_STRING) -> None:
    """Refuse a row or document with a text field or a long string.

    Args:
        obj: Parsed JSON.
        where: Location for the error message.
        limit: Longest allowed string.

    Raises:
        UnsafeExportError: If it could carry dataset text.
    """
    if (field := text_field(obj)) is not None:
        raise UnsafeExportError(f"{where}: text field {field!r}")
    if (s := long_string(obj, limit)) is not None:
        raise UnsafeExportError(f"{where}: string of {len(s)} characters")


def check_row(row: Mapping[str, Any], where: str, limit: int = MAX_STRING) -> None:
    """Refuse a row with a text field (at any depth) or a long string.

    Args:
        row: One parsed row.
        where: Location for the error message.
        limit: Longest allowed string.

    Raises:
        UnsafeExportError: If the row could carry dataset text.
    """
    check_document(dict(row), where, limit)


def check_rows(path: Path, limit: int = MAX_STRING) -> int:
    """Count the rows of a JSONL file, refusing any that could carry text.

    Args:
        path: A JSONL file.
        limit: Longest allowed string.

    Returns:
        Number of rows.

    Raises:
        UnsafeExportError: If a row has a text field or a long string, or a
            line is not a JSON object.
    """
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise UnsafeExportError(f"{path.name}: row {n} is not an object")
            check_row(row, f"{path.name} row {n}", limit)
            n += 1
    return n


def check_npz(path: Path, limit: int = MAX_STRING) -> None:
    """Refuse saved arrays that hold objects or long strings.

    Args:
        path: An ``.npz`` file (loaded without pickle).
        limit: Longest allowed string.

    Raises:
        UnsafeExportError: If an array is not numeric, boolean or short
            strings.
    """
    try:
        z = np.load(path, allow_pickle=False)
    except ValueError as err:
        raise UnsafeExportError(f"{path.name}: {err}") from err
    with z:
        for name in z.files:
            try:
                arr = z[name]
            except ValueError as err:  # object arrays need pickle
                raise UnsafeExportError(f"{path.name}:{name}: {err}") from err
            if arr.dtype.kind in "biuf":
                continue
            if arr.dtype.kind != "U":
                raise UnsafeExportError(f"{path.name}:{name}: dtype {arr.dtype}")
            if arr.size and int(np.char.str_len(arr).max()) > limit:
                raise UnsafeExportError(f"{path.name}:{name}: long strings")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Args:
        path: File.

    Returns:
        Hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select(output_dir: Path, names: Iterable[str]) -> list[Path]:
    """List the exportable files of some experiments.

    Args:
        output_dir: Root of the outputs.
        names: Experiment or analysis names (output subdirectories).

    Returns:
        Sorted paths relative to ``output_dir``.
    """
    out: set[Path] = set()
    for name in names:
        root = output_dir / name
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.is_file() and not p.is_symlink() and p.suffix in SUFFIXES:
                out.add(p.relative_to(output_dir))
    return sorted(out)


def readme(files: Sequence[Mapping[str, Any]]) -> str:
    """Return the export's README.

    Args:
        files: Manifest entries.

    Returns:
        Markdown text.
    """
    rows = sum(int(f.get("rows") or 0) for f in files)
    return "\n".join(
        [
            "# beyond-answer-confidence results export",
            "",
            f"Built with beyond-answer-confidence {__version__}.",
            "",
            f"- {len(files)} files, {rows:,} JSONL rows; SHA-256 of every file"
            f" in `{MANIFEST}`.",
            "- Rows hold item ids, option letters or labels, correctness and the",
            "  backend's probabilities. No dataset text is included: every row",
            "  was checked for text fields and long strings before export.",
            "- Request caches are not included, because they contain the full",
            "  prompts. The prompts can be rebuilt from the pinned dataset",
            "  revisions in `DATASETS.md` with the package's deterministic item",
            "  builders (`beyond-answer-confidence verify` checks the rebuild).",
            "- Rows derived from a dataset inherit its licence. Rows from",
            "  non-commercial datasets (CC BY-NC, see `DATASETS.md`) may be used",
            "  for non-commercial purposes only.",
            "- `*.json` result files are copied with internal (`_`-prefixed)",
            "  fields removed; `*.jsonl` row files are gzipped.",
            "- `distractors/`, if present, holds nearest-neighbour distractor",
            "  choices as integer positions (no text); copy it to",
            "  `<data_dir>/cache/distractors/` to rebuild prompts independently",
            "  of floating-point details of the embedding model.",
            "",
        ]
    )


def datasets_md() -> str:
    """Return the dataset table as Markdown.

    Returns:
        The table.
    """
    head = "| Dataset | Source | Pinned revision | Licence |\n|---|---|---|---|\n"
    body = "".join(
        f"| {d.name} | {d.source} | `{d.revision}` | {d.licence} |\n" for d in DATASETS
    )
    note = (
        "\nDatasets are downloaded at run time from their sources and remain"
        " under their own licences.\n"
    )
    return "# Datasets\n\n" + head + body + note


def is_previous_export(out: Path) -> bool:
    """Return whether a directory holds an export written by this package.

    Args:
        out: A directory.

    Returns:
        True if its manifest carries :data:`MANIFEST_SENTINEL`.
    """
    manifest = out / MANIFEST
    if not manifest.is_file() or manifest.is_symlink():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and all(
        data.get(k) == v for k, v in MANIFEST_SENTINEL.items()
    )


def _prepare(out: Path) -> None:
    """Create an empty export directory, replacing only a previous export."""
    if out.is_symlink():
        raise FileExistsError(f"{out} is a symbolic link; refusing to use it")
    if out.exists():
        if not out.is_dir():
            raise FileExistsError(f"{out} exists and is not a directory")
        if any(out.iterdir()):
            if not is_previous_export(out):
                raise FileExistsError(
                    f"{out} is not empty and is not a previous beyond-answer-confidence "
                    "export; refusing to overwrite it"
                )
            shutil.rmtree(out)
        else:
            out.rmdir()
    out.mkdir(parents=True)


def _entry(dst: Path, out: Path, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "path": dst.relative_to(out).as_posix(),
        "bytes": dst.stat().st_size,
        "sha256": sha256(dst),
    }


def check_file(src: Path, limit: int = MAX_STRING) -> None:
    """Refuse an output file that could carry dataset text.

    Args:
        src: A ``.jsonl``, ``.json`` or ``.npz`` file.
        limit: Longest allowed string.

    Raises:
        UnsafeExportError: If the file could carry dataset text.
    """
    if src.suffix == ".jsonl":
        check_rows(src, limit)
    elif src.suffix == ".json":
        data = strip_private(json.loads(src.read_text(encoding="utf-8")))
        check_document(data, src.name, limit)
    else:
        check_npz(src, limit)


def _copy_one(src: Path, rel: Path, out: Path, limit: int) -> dict[str, Any]:
    """Copy one checked output file; return its manifest entry."""
    extra: dict[str, Any] = {}
    if src.suffix == ".jsonl":
        extra["rows"] = check_rows(src, limit)
        dst = out / rel.with_name(rel.name + ".gz")
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Rows are re-serialised strictly (non-finite floats as null).
        with (
            src.open(encoding="utf-8") as fin,
            gzip.GzipFile(dst, "wb", mtime=0) as fout,
        ):
            for line in fin:
                if line.strip():
                    row = json.loads(line)
                    fout.write((dumps(row, ensure_ascii=False) + "\n").encode())
    elif src.suffix == ".json":
        data = strip_private(json.loads(src.read_text(encoding="utf-8")))
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(dumps(data, indent=1, default=str), encoding="utf-8")
    else:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    return _entry(dst, out, rel.as_posix()) | extra


def build(
    output_dir: Path,
    names: Sequence[str],
    out: Path,
    *,
    distractors: Path | None = None,
    limit: int = MAX_STRING,
) -> dict[str, Any]:
    """Write the export directory.

    Every file is checked before anything is written, so an unsafe file
    leaves no partial export behind.

    Args:
        output_dir: Root of the outputs.
        names: Experiments and analyses to include.
        out: Export directory (created; a previous export is replaced).
        distractors: Directory of distractor-choice JSON files to include.
        limit: Longest allowed string in rows and result files.

    Returns:
        The manifest.

    Raises:
        UnsafeExportError: If any file could carry dataset text.
        ValueError: If ``out`` lies inside an exported directory.
    """
    out_r = out.resolve()
    for name in names:
        if out_r.is_relative_to((output_dir / name).resolve()):
            raise ValueError(f"export directory {out} is inside {output_dir / name}")
    files = select(output_dir, names)
    extra = sorted(distractors.glob("*.json")) if distractors is not None else []
    for rel in files:  # check everything first
        check_file(output_dir / rel, limit)
    for src in extra:
        check_document(
            json.loads(src.read_text(encoding="utf-8")),
            f"distractors/{src.name}",
            limit,
        )
    _prepare(out)
    entries = [_copy_one(output_dir / rel, rel, out, limit) for rel in files]
    for src in extra:
        dst = out / "distractors" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        entries.append(_entry(dst, out, f"distractors/{src.name}"))
    manifest = {
        **MANIFEST_SENTINEL,
        "package_version": __version__,
        "experiments": list(names),
        "files": entries,
        "datasets": [d.__dict__ for d in DATASETS],
    }
    (out / MANIFEST).write_text(dumps(manifest, indent=1), encoding="utf-8")
    (out / "README.md").write_text(readme(entries), encoding="utf-8")
    (out / "DATASETS.md").write_text(datasets_md(), encoding="utf-8")
    logger.info("wrote %s (%d files)", out, len(entries))
    return manifest


def make_tar(out: Path, *, force: bool = False) -> Path:
    """Write ``<out>.tar.gz`` next to an export directory.

    Args:
        out: Export directory.
        force: Replace an existing archive.

    Returns:
        The archive path.

    Raises:
        FileExistsError: If the archive exists and ``force`` is not set.
    """
    tar_path = out.with_name(out.name + ".tar.gz")
    if tar_path.exists() and not force:
        raise FileExistsError(f"{tar_path} exists; pass --force to replace it")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out, arcname=out.name)
    logger.info("wrote %s", tar_path)
    return tar_path
