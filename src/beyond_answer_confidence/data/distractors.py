"""Wrong answer options: sampled, perturbed, or nearest in embedding space.

:func:`sample_distractors` and :func:`perturb_answer` are pure.
:func:`nearest_distractors` embeds answers with a small local sentence
encoder (needs the ``models`` extra) and saves its choice under
``<data_dir>/cache/distractors/`` so later builds are exact and need no
model.
"""

import hashlib
import json
import random
import re
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import numpy as np

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
"""Sentence encoder for nearest distractors."""

ENCODER_REVISION = (
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # pragma: allowlist secret
)
"""Pinned encoder commit."""

DistractorPicker = Callable[
    [Sequence[str], Sequence[Sequence[str]], Sequence[str]], list[list[str] | None]
]
"""``(golds, aliases_per_gold, pool) -> distractors per gold (None = too few)``."""

_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def normalise_answer(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace (for comparisons).

    Args:
        text: An answer string.

    Returns:
        The normalised string.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def sample_distractors(
    gold: str,
    pool: Sequence[str],
    exclude: Sequence[str],
    k: int,
    rng: random.Random,
) -> list[str] | None:
    """Draw ``k`` distinct distractors from a pool, avoiding the gold's aliases.

    Candidates are de-duplicated after normalisation (first spelling kept, in
    pool order) before one ``rng.sample`` call.

    Args:
        gold: Correct answer.
        pool: Candidate answers (e.g. other objects of the same relation).
        exclude: Aliases of the gold answer.
        k: Number of distractors.
        rng: Random generator.

    Returns:
        The distractors, or ``None`` if the pool is too small.
    """
    banned = {normalise_answer(gold), *(normalise_answer(a) for a in exclude)}
    seen: set[str] = set()
    candidates = []
    for c in pool:
        n = normalise_answer(c)
        if n and n not in banned and n not in seen:
            seen.add(n)
            candidates.append(c)
    if len(candidates) < k:
        return None
    return rng.sample(candidates, k)


def perturb_answer(answer: str, rng: random.Random, k: int = 3) -> list[str] | None:
    """Make ``k`` wrong answers of the same form by changing a year or number.

    A four-digit year is shifted by distinct offsets in +/-1..6; otherwise
    the first number is scaled by distinct factors, keeping its formatting.

    Args:
        answer: Correct answer (a date or a number).
        rng: Random generator.
        k: Number of variants.

    Returns:
        Variants, or ``None`` if the answer contains no number or the
        variants are not distinct.
    """
    m = _YEAR.search(answer)
    if m:
        year = int(m.group())
        offsets = rng.sample([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6], k)
        return [
            answer[: m.start()] + str(year + o) + answer[m.end() :] for o in offsets
        ]
    m = _NUMBER.search(answer)
    if not m:
        return None
    raw = m.group()
    value = float(raw.replace(",", ""))
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    factors = rng.sample([0.5, 0.7, 0.8, 1.25, 1.5, 2.0, 3.0], k)
    variants = []
    for f in factors:
        new = value * f
        if decimals == 0:
            new_i = round(new)
            if new_i == int(value):
                new_i += 1
            text = f"{new_i:,}" if "," in raw else str(new_i)
        else:
            text = f"{new:.{decimals}f}"
        variants.append(answer[: m.start()] + text + answer[m.end() :])
    return variants if len(set(variants)) == k else None


def embed(texts: Sequence[str], data_dir: Path, batch: int = 256) -> np.ndarray:
    """Mean-pooled, L2-normalised sentence embeddings (local CPU).

    Args:
        texts: Strings to embed.
        data_dir: Data directory (model cache under ``data_dir/hf``).
        batch: Batch size.

    Returns:
        ``(len(texts), 384)`` array.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    cache = str(data_dir / "hf")
    tok = AutoTokenizer.from_pretrained(
        ENCODER, revision=ENCODER_REVISION, cache_dir=cache
    )
    # The pinned revision ships model.safetensors; never fall back to pickle.
    model = AutoModel.from_pretrained(
        ENCODER, revision=ENCODER_REVISION, cache_dir=cache, use_safetensors=True
    )
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(
                list(texts[i : i + batch]),
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(vec, dim=-1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 384))


def index_key(
    golds: Sequence[str],
    excludes: Sequence[Sequence[str]],
    unique_pool: Sequence[str],
    k: int,
    max_sim: float,
) -> str:
    """Hash every input of a nearest-distractor choice (the saved file's name).

    The format is fixed: changing it would orphan the saved choices.

    Args:
        golds: Correct answers.
        excludes: Aliases per gold.
        unique_pool: De-duplicated candidate pool.
        k: Distractors per gold.
        max_sim: Near-duplicate similarity cut.

    Returns:
        A hex SHA-256 digest.
    """
    return hashlib.sha256(
        json.dumps(
            [
                ENCODER,
                ENCODER_REVISION,
                k,
                max_sim,
                list(golds),
                [list(e) for e in excludes],
                list(unique_pool),
            ],
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def nearest_distractors(
    golds: Sequence[str],
    excludes: Sequence[Sequence[str]],
    pool: Sequence[str],
    data_dir: Path,
    k: int = 3,
    max_sim: float = 0.92,
) -> list[list[str] | None]:
    """Pick the ``k`` pool answers most similar to each gold (plausible distractors).

    Near-duplicates of the gold (cosine > ``max_sim``), its aliases, and
    candidates containing (or contained in) the gold or an alias are
    skipped, as are distractors equal to each other after normalisation.

    The choice is saved under ``data_dir/cache/distractors/`` as integer
    positions in the de-duplicated pool, keyed by :func:`index_key`, and
    reused on later calls. Embeddings differ in the last floating-point
    digits with the number of CPU threads, which can swap near-tied
    neighbours; the saved positions make rebuilds exact. They contain no
    dataset text.

    Args:
        golds: Correct answers.
        excludes: Aliases per gold.
        pool: Candidate answers.
        data_dir: Data directory.
        k: Distractors per gold.
        max_sim: Similarity above which a candidate counts as the same answer.

    Returns:
        Distractors per gold (``None`` if fewer than ``k`` qualify).
    """
    uniq = list(dict.fromkeys(p for p in pool if normalise_answer(p)))
    key = index_key(golds, excludes, uniq, k, max_sim)
    saved = data_dir / "cache" / "distractors" / f"{key}.json"
    if saved.exists():
        picks = json.loads(saved.read_text(encoding="utf-8"))["picks"]
        return [None if p is None else [uniq[j] for j in p] for p in picks]
    chosen = nearest_indices(
        golds,
        excludes,
        uniq,
        embed(uniq, data_dir),
        embed(list(golds), data_dir),
        k,
        max_sim,
    )
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(
        json.dumps({"encoder": ENCODER_REVISION, "k": k, "picks": chosen}),
        encoding="utf-8",
    )
    return [None if p is None else [uniq[j] for j in p] for p in chosen]


def nearest_indices(
    golds: Sequence[str],
    excludes: Sequence[Sequence[str]],
    uniq: Sequence[str],
    pool_vec: np.ndarray,
    gold_vec: np.ndarray,
    k: int,
    max_sim: float,
) -> list[list[int] | None]:
    """Positions in ``uniq`` of each gold's ``k`` nearest acceptable distractors.

    Args:
        golds: Correct answers.
        excludes: Aliases per gold.
        uniq: De-duplicated candidate pool.
        pool_vec: Unit-norm embeddings of ``uniq``.
        gold_vec: Unit-norm embeddings of ``golds``.
        k: Distractors per gold.
        max_sim: Near-duplicate similarity cut.

    Returns:
        Positions per gold (``None`` if fewer than ``k`` qualify).
    """
    out: list[list[int] | None] = []
    block = np.zeros((0, len(uniq)))
    for gi, (gold, exc) in enumerate(zip(golds, excludes, strict=True)):
        if gi % 512 == 0:  # similarities in blocks: the full matrix can be GBs
            block = gold_vec[gi : gi + 512] @ pool_vec.T
        row = block[gi % 512]
        banned = {normalise_answer(gold), *(normalise_answer(a) for a in exc)}
        picked: list[int] = []
        seen: set[str] = set()
        m = min(200, len(row) - 1)
        top = np.argpartition(-row, m)[:m]
        for j in top[np.argsort(-row[top])]:
            n = normalise_answer(uniq[j])
            if row[j] > max_sim or n in banned or n in seen:
                continue
            if any(n in b or b in n for b in banned if len(b) > 3 and len(n) > 3):
                continue
            picked.append(int(j))
            seen.add(n)
            if len(picked) == k:
                break
        out.append(picked if len(picked) == k else None)
    return out


def distractor_picker(data_dir: Path, k: int = 3) -> DistractorPicker:
    """Bind :func:`nearest_distractors` to a data directory.

    Args:
        data_dir: Data directory.
        k: Distractors per gold.

    Returns:
        A picker for the item builders.
    """
    return partial(nearest_distractors, data_dir=data_dir, k=k)
