"""Banking77 and CLINC150 intent-classification data with a common shape.

Both sources are pinned to a fixed revision so the data cannot change under
an analysis. Downloads land under ``<data_dir>/raw`` and ``<data_dir>/hf``;
keep that directory out of version control.

Licences: Banking77 is CC BY 4.0, CLINC150 is CC BY 3.0.
"""

import csv
import io
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from beyond_answer_confidence.data.download import Fetcher, pinned_file

# The Hugging Face repository of Banking77 only has a loading script, which
# current ``datasets`` versions no longer execute; it downloads these CSVs, so
# they are fetched directly at a pinned commit.
BANKING77_COMMIT = (
    "57ec275d8078af65b7731c2a98be812d844a6d6b"  # pragma: allowlist secret
)
"""Public git commit of the Banking77 CSV files."""
BANKING77_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "{commit}/banking_data/{split}.csv"
)
BANKING77_SHA256 = {
    "train": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",  # pragma: allowlist secret
    "test": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",  # pragma: allowlist secret
}
"""SHA-256 of each Banking77 CSV at :data:`BANKING77_COMMIT`."""
CLINC150_REVISION = (
    "155b9c710419136e17307b80d0a13e68cd46b4ec"  # pragma: allowlist secret
)
"""Public Hugging Face revision of ``clinc/clinc_oos``."""
CLINC150_OOS_LABEL = "oos"

DEV_PER_INTENT = 5
"""Banking77 train examples per intent held out as a development set."""

DATASETS = ("banking77", "clinc150")


@dataclass(frozen=True)
class Example:
    """One labelled utterance.

    Attributes:
        text: The utterance.
        label: Its intent.
    """

    text: str
    label: str


@dataclass(frozen=True)
class IntentDataset:
    """An intent-classification dataset.

    Attributes:
        name: Short dataset name.
        labels: In-scope intent names, sorted.
        pool: Train examples usable as in-context or training examples.
        dev: Held-out development examples; disjoint from ``pool``.
        test: In-scope test examples.
        oos_test: Out-of-scope test examples (empty for Banking77).
        oos_pool: Out-of-scope train examples, used as irrelevant filler.
    """

    name: str
    labels: tuple[str, ...]
    pool: tuple[Example, ...]
    dev: tuple[Example, ...]
    test: tuple[Example, ...]
    oos_test: tuple[Example, ...] = ()
    oos_pool: tuple[Example, ...] = ()

    def examples_by_label(self) -> dict[str, list[Example]]:
        """Group the example pool by intent.

        Returns:
            Pool examples keyed by label, in pool order.
        """
        grouped: dict[str, list[Example]] = defaultdict(list)
        for ex in self.pool:
            grouped[ex.label].append(ex)
        return dict(grouped)

    def split(self, name: str) -> tuple[Example, ...]:
        """Return the in-scope examples of a split.

        Args:
            name: ``"dev"`` or ``"test"``.

        Returns:
            The examples.

        Raises:
            ValueError: For another split name.
        """
        if name == "dev":
            return self.dev
        if name == "test":
            return self.test
        raise ValueError(f"unknown split {name!r}")


def split_dev(
    train: Iterable[Example], per_intent: int = DEV_PER_INTENT, seed: int = 0
) -> tuple[tuple[Example, ...], tuple[Example, ...]]:
    """Hold out ``per_intent`` random train examples per intent as a dev set.

    Args:
        train: Training examples.
        per_intent: Examples per intent moved to the dev set.
        seed: Seed for the per-intent shuffle.

    Returns:
        ``(pool, dev)``, both in a deterministic order.
    """
    grouped: dict[str, list[Example]] = defaultdict(list)
    for ex in train:
        grouped[ex.label].append(ex)
    # Data split, not crypto.
    rng = random.Random(seed)  # noqa: S311  # nosec B311
    pool: list[Example] = []
    dev: list[Example] = []
    for label in sorted(grouped):
        items = grouped[label][:]
        rng.shuffle(items)
        dev.extend(items[:per_intent])
        pool.extend(items[per_intent:])
    return tuple(pool), tuple(dev)


def parse_banking77_csv(content: str) -> list[Example]:
    """Parse a Banking77 CSV with ``text,category`` columns.

    Args:
        content: The CSV file contents.

    Returns:
        The examples in file order.
    """
    return [
        Example(row["text"], row["category"])
        for row in csv.DictReader(io.StringIO(content))
    ]


def load_banking77(data_dir: Path, fetcher: Fetcher | None = None) -> IntentDataset:
    """Load Banking77 (77 intents) at the pinned commit.

    The CSVs are downloaded once and verified against
    :data:`BANKING77_SHA256` (see :func:`~beyond_answer_confidence.data.download.pinned_file`).

    Args:
        data_dir: Root data directory.
        fetcher: Downloads a URL (default: an https download).

    Returns:
        The dataset; its dev set is carved from train.
    """
    splits = {}
    for split in ("train", "test"):
        path = pinned_file(
            BANKING77_URL.format(commit=BANKING77_COMMIT, split=split),
            data_dir / "raw" / "banking77" / BANKING77_COMMIT / f"{split}.csv",
            BANKING77_SHA256[split],
            fetcher,
        )
        splits[split] = parse_banking77_csv(path.read_text(encoding="utf-8"))
    pool, dev = split_dev(splits["train"])
    labels = tuple(sorted({ex.label for ex in splits["train"]}))
    return IntentDataset("banking77", labels, pool, dev, tuple(splits["test"]))


def load_clinc150(data_dir: Path) -> IntentDataset:
    """Load CLINC150 "plus" (150 intents plus out-of-scope) at the pinned revision.

    Args:
        data_dir: Root data directory (``<data_dir>/hf`` is the download cache).

    Returns:
        The dataset with out-of-scope examples separated out. Its dev set is
        the in-scope validation split.
    """
    from datasets import load_dataset

    # Pinned revision (satisfies bandit B615).
    ds = load_dataset(
        "clinc/clinc_oos",
        "plus",
        revision=CLINC150_REVISION,
        cache_dir=str(data_dir / "hf"),
    )
    names: list[str] = ds["train"].features["intent"].names

    def convert(split: str) -> list[Example]:
        return [Example(row["text"], names[row["intent"]]) for row in ds[split]]

    def in_scope(examples: list[Example]) -> tuple[Example, ...]:
        return tuple(ex for ex in examples if ex.label != CLINC150_OOS_LABEL)

    def oos(examples: list[Example]) -> tuple[Example, ...]:
        return tuple(ex for ex in examples if ex.label == CLINC150_OOS_LABEL)

    train, val, test = convert("train"), convert("validation"), convert("test")
    labels = tuple(sorted(n for n in names if n != CLINC150_OOS_LABEL))
    return IntentDataset(
        "clinc150",
        labels,
        pool=in_scope(train),
        dev=in_scope(val),
        test=in_scope(test),
        oos_test=oos(test),
        oos_pool=oos(train),
    )


def load(name: str, data_dir: Path) -> IntentDataset:
    """Load a dataset by name.

    Args:
        name: ``"banking77"`` or ``"clinc150"``.
        data_dir: Root data directory.

    Returns:
        The dataset.

    Raises:
        ValueError: For an unknown name.
    """
    if name == "banking77":
        return load_banking77(data_dir)
    if name == "clinc150":
        return load_clinc150(data_dir)
    raise ValueError(f"unknown dataset {name!r}")


def filler_texts(datasets: dict[str, IntentDataset], data_dir: Path) -> list[str]:
    """Out-of-scope utterances used as irrelevant filler (from CLINC150).

    Args:
        datasets: Already loaded datasets (CLINC150 is reused if present).
        data_dir: Root data directory, to load CLINC150 otherwise.

    Returns:
        The CLINC150 out-of-scope train utterances.
    """
    clinc = datasets.get("clinc150") or load_clinc150(data_dir)
    return [ex.text for ex in clinc.oos_pool]


def select_items(
    examples: Sequence[Example], per_intent: int, seed: int = 0
) -> list[int]:
    """Stratified random subsample: ``per_intent`` indices per label.

    The order within each label is random but fixed by ``seed``, so the first
    k of each label form a nested subset.

    Args:
        examples: The split to sample from.
        per_intent: Items per label (all items if a label has fewer).
        seed: Sampling seed.

    Returns:
        Selected indices, grouped by label in sorted label order.
    """
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        by_label[ex.label].append(i)
    # Sampling, not crypto.
    rng = random.Random(seed)  # noqa: S311  # nosec B311
    out: list[int] = []
    for label in sorted(by_label):
        idx = by_label[label][:]
        rng.shuffle(idx)
        out.extend(idx[:per_intent])
    return out


def spread_indices(size: int, n: int) -> list[int]:
    """``n`` indices spread evenly through a sequence (e.g. across an intent-ordered split).

    Args:
        size: Sequence length.
        n: Number of indices.

    Returns:
        Distinct sorted indices.
    """
    return sorted({i * size // n for i in range(min(n, size))})


def nested_subset(
    examples: Sequence[Example], selected: Sequence[int], k: int
) -> list[int]:
    """First ``k`` selected items of each label.

    Args:
        examples: The split.
        selected: Output of :func:`select_items`.
        k: Items per label.

    Returns:
        The subset, preserving order.
    """
    seen: dict[str, int] = defaultdict(int)
    out = []
    for i in selected:
        label = examples[i].label
        if seen[label] < k:
            out.append(i)
            seen[label] += 1
    return out
