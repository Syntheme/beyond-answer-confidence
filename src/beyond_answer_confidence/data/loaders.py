"""Loaders for the public question-answering and NLI datasets, at pinned revisions.

Every source is fixed to a commit so the data cannot change under a run.
Downloads go to ``<data_dir>/hf`` (Hugging Face cache) and ``<data_dir>/raw``
(other files); both hold dataset text and must stay out of version control.
Each loader returns the file as a DataFrame with no filtering beyond what its
docstring says; item builders in :mod:`beyond_answer_confidence.tasks` do the rest.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.data.download import Fetcher, pinned_file

# Public Hugging Face / GitHub commit hashes (not secrets).
REVISIONS = {
    "akariasai/PopQA": "098765c79ea10a2cb19c828324e33281b8336ec0",  # pragma: allowlist secret
    "agentic-learning-ai-lab/daily-oracle": "455b35b2a2415034b05ef91e56c80191fef4a737",  # pragma: allowlist secret
    "hotpotqa/hotpot_qa": "1908d6afbbead072334abe2965f91bd2709910ab",  # pragma: allowlist secret
    "community-datasets/qanta": "e3c5602229e5c0c8572636a81cb366eb9ccfa890",  # pragma: allowlist secret
    "ShuoZheLi/SelfAware": "ffdfba5ce6a190e3a29f0497fcea77d9620eb4cc",  # pragma: allowlist secret
    "google/simpleqa-verified": "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc",  # pragma: allowlist secret
    "truthfulqa/truthful_qa": "741b8276f2d1982aa3d5b832d3ee81ed3b896490",  # pragma: allowlist secret
    "mandarjoshi/trivia_qa": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",  # pragma: allowlist secret
    "sewon/ambig_qa": "e969d0132f4dd28c2939d55be34f1788c00ccfe7",  # pragma: allowlist secret
    "earino/chaosnli": "bf8d66a1d3434ac8b90ffc392b1b5e44bdf7e439",  # pragma: allowlist secret
}
"""Dataset repo -> pinned commit."""

TRUTHFULQA_CSV_COMMIT = (
    "f6be04e52bbcb41d4d20daee6358231d4a5015d2"  # pragma: allowlist secret
)
"""Commit of the official TruthfulQA CSV (it has the best / best-incorrect
answer columns that the Hub copy lacks)."""

TRUTHFULQA_CSV_URL = (
    "https://raw.githubusercontent.com/sylinrl/TruthfulQA/{commit}/TruthfulQA.csv"
)
TRUTHFULQA_CSV_SHA256 = "b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c"  # pragma: allowlist secret
"""SHA-256 of the TruthfulQA CSV at :data:`TRUTHFULQA_CSV_COMMIT`."""

CHAOSNLI_SUBSETS = ("snli", "mnli_m", "alphanli")
"""ChaosNLI parts."""


def hf_file(repo: str, filename: str, data_dir: Path) -> Path:
    """Download one file of a pinned dataset repo (cached).

    Args:
        repo: Dataset repo id (must be in :data:`REVISIONS`).
        filename: Path inside the repo.
        data_dir: Data directory; files are cached under ``data_dir/hf``.

    Returns:
        Local path.
    """
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo,
            filename,
            repo_type="dataset",
            revision=REVISIONS[repo],
            cache_dir=str(data_dir / "hf"),
        )
    )


def load_popqa(data_dir: Path) -> pd.DataFrame:
    """PopQA (14,267 questions with subject popularity).

    Args:
        data_dir: Data directory.

    Returns:
        One row per question; ``possible_answers`` parsed from JSON.
    """
    df = pd.read_csv(
        hf_file("akariasai/PopQA", "test.tsv", data_dir),
        sep="\t",
        keep_default_na=False,
    )
    df["possible_answers"] = df["possible_answers"].map(json.loads)
    return df


def load_daily_oracle(kind: str, data_dir: Path) -> pd.DataFrame:
    """Daily Oracle news questions, 2020-01 to 2026-07.

    Args:
        kind: ``"tf"`` (yes/no) or ``"mc"`` (four options).
        data_dir: Data directory.

    Returns:
        One row per question with a ``month`` column (``YYYY-MM``).
    """
    name = f"{kind}_questions_2020-01-01_2026-07-18.csv"
    df = pd.read_csv(
        hf_file("agentic-learning-ai-lab/daily-oracle", name, data_dir),
        keep_default_na=False,
    )
    df["month"] = df["date"].str[:7]
    return df


def load_hotpot_validation(data_dir: Path) -> pd.DataFrame:
    """HotpotQA distractor-setting validation split (7,405 questions).

    Args:
        data_dir: Data directory.

    Returns:
        The split.
    """
    return pd.read_parquet(
        hf_file(
            "hotpotqa/hotpot_qa",
            "distractor/validation-00000-of-00001.parquet",
            data_dir,
        )
    )


def load_qanta(fold: str, data_dir: Path) -> pd.DataFrame:
    """Quizbowl (QANTA) questions split into sentences.

    Args:
        fold: e.g. ``"guessdev"``.
        data_dir: Data directory.

    Returns:
        One row per sentence (``qanta_id``, ``id``, ``text``, ``page``,
        ``category``, ...).
    """
    return pd.read_parquet(
        hf_file(
            "community-datasets/qanta",
            f"mode=sentences,char_skip=25/{fold}-00000-of-00001.parquet",
            data_dir,
        )
    )


def load_selfaware(data_dir: Path) -> pd.DataFrame:
    """SelfAware (3,369 questions, 1,032 unanswerable).

    Args:
        data_dir: Data directory.

    Returns:
        The questions.
    """
    raw = json.loads(
        hf_file("ShuoZheLi/SelfAware", "SelfAware.json", data_dir).read_text()
    )
    return pd.DataFrame(raw["example"])


def load_simpleqa(data_dir: Path) -> pd.DataFrame:
    """SimpleQA Verified (1,000 questions).

    Args:
        data_dir: Data directory.

    Returns:
        The questions.
    """
    return pd.read_csv(
        hf_file("google/simpleqa-verified", "simpleqa_verified.csv", data_dir),
        keep_default_na=False,
    )


def load_truthfulqa_binary(
    data_dir: Path, fetcher: Fetcher | None = None
) -> pd.DataFrame:
    """TruthfulQA, official CSV (Best Answer / Best Incorrect Answer; 790 rows).

    The file is downloaded once to ``data_dir/raw`` and verified against
    :data:`TRUTHFULQA_CSV_SHA256` (see :func:`~beyond_answer_confidence.data.download.pinned_file`).

    Args:
        data_dir: Data directory.
        fetcher: Downloads a URL (default: an https download).

    Returns:
        The questions.
    """
    path = pinned_file(
        TRUTHFULQA_CSV_URL.format(commit=TRUTHFULQA_CSV_COMMIT),
        data_dir / "raw" / f"TruthfulQA_{TRUTHFULQA_CSV_COMMIT[:12]}.csv",
        TRUTHFULQA_CSV_SHA256,
        fetcher,
    )
    return pd.read_csv(path, keep_default_na=False)


def load_truthfulqa_mc1(data_dir: Path) -> pd.DataFrame:
    """TruthfulQA ``multiple_choice`` (mc1 targets, 817 questions).

    Args:
        data_dir: Data directory.

    Returns:
        The questions.
    """
    return pd.read_parquet(
        hf_file(
            "truthfulqa/truthful_qa",
            "multiple_choice/validation-00000-of-00001.parquet",
            data_dir,
        )
    )


def load_triviaqa_validation(data_dir: Path) -> pd.DataFrame:
    """TriviaQA ``rc.nocontext`` validation (17,944 rows, 9,960 questions).

    Args:
        data_dir: Data directory.

    Returns:
        The rows (a question repeats once per evidence document).
    """
    return pd.read_parquet(
        hf_file(
            "mandarjoshi/trivia_qa",
            "rc.nocontext/validation-00000-of-00001.parquet",
            data_dir,
        )
    )


def load_ambigqa_validation(data_dir: Path) -> pd.DataFrame:
    """AmbigQA ``light`` validation (2,002 questions).

    Args:
        data_dir: Data directory.

    Returns:
        The questions.
    """
    return pd.read_parquet(
        hf_file("sewon/ambig_qa", "light/validation-00000-of-00001.parquet", data_dir)
    )


def load_chaosnli(subset: str, data_dir: Path) -> pd.DataFrame:
    """ChaosNLI with 100 annotator labels per item.

    Args:
        subset: One of :data:`CHAOSNLI_SUBSETS`.
        data_dir: Data directory.

    Returns:
        One row per item.
    """
    path = hf_file("earino/chaosnli", f"raw/chaosNLI_{subset}.jsonl", data_dir)
    return pd.read_json(path, lines=True)


def to_jsonable(value: Any) -> Any:
    """Convert numpy containers (as read from parquet) to plain Python.

    Args:
        value: Any value.

    Returns:
        Lists, dicts and scalars only.
    """
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def table_rows(df: pd.DataFrame) -> Iterator[Any]:
    """Iterate over a table's rows as named tuples (``DataFrame.itertuples``).

    Dataset columns hold strings, numbers, arrays and dicts alike, so the
    rows are typed as ``Any``; builders convert the fields they use.

    Args:
        df: A table.

    Returns:
        An iterator over the rows (the index comes first, as ``Index``).
    """
    rows: Iterator[Any] = df.itertuples()
    return rows
