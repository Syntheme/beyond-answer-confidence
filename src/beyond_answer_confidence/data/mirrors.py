"""Check the two unofficial dataset mirrors against independent sources.

ChaosNLI is loaded from the Hugging Face mirror ``earino/chaosnli`` and
SelfAware from ``ShuoZheLi/SelfAware``. :func:`check_mirrors` compares them
with independent copies at pinned commits (public files only, no API calls):

- **SelfAware**: the authors' GitHub release; every example is compared
  field by field.
- **ChaosNLI**: the authors' original download link no longer works and no
  other full copy is public. The MNLI part is compared item by item with
  ``tasksource/chaos-mnli-ambiguity``; for all three parts the item counts
  are compared with the published counts (Nie et al., 2020).
"""

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from beyond_answer_confidence.data import loaders
from beyond_answer_confidence.data.download import Fetcher, fetch

logger = logging.getLogger(__name__)

SELFAWARE_GITHUB = (
    "https://raw.githubusercontent.com/yinzhangyue/SelfAware/"
    "f0bad1ff77bd42fc4eb2360281ed646c7bb7bd0c/data/SelfAware.json"  # pragma: allowlist secret
)
"""The authors' SelfAware release at a pinned commit."""

TASKSOURCE_MNLI = (
    "https://huggingface.co/datasets/tasksource/chaos-mnli-ambiguity/resolve/"
    "64e9c55ec72f36df266479ac200069cc3613d2ff/chaos_mnli.jsonl"  # pragma: allowlist secret
)
"""An independent copy of ChaosNLI-MNLI at a pinned commit."""

PUBLISHED_COUNTS = {"snli": 1514, "mnli_m": 1599, "alphanli": 1532}
"""ChaosNLI item counts as published by its authors (Nie et al., 2020)."""


def compare_selfaware(
    mirror: Sequence[Mapping[Any, Any]], original: Sequence[Mapping[Any, Any]]
) -> dict[str, Any]:
    """Compare SelfAware examples by question id.

    Args:
        mirror: Examples from the mirror.
        original: Examples from the authors' release.

    Returns:
        Counts of shared, missing and differing examples, and up to five
        differing ids.
    """
    m = {str(e["question_id"]): e for e in mirror}
    o = {str(e["question_id"]): e for e in original}
    shared = m.keys() & o.keys()
    differ = sorted(q for q in shared if m[q] != o[q])
    return {
        "mirror": len(m),
        "original": len(o),
        "shared": len(shared),
        "only_in_mirror": len(m.keys() - o.keys()),
        "only_in_original": len(o.keys() - m.keys()),
        "differing": len(differ),
        "differing_examples": differ[:5],
    }


def compare_chaos_mnli(
    mirror: Sequence[Mapping[Any, Any]], other: Sequence[Mapping[Any, Any]]
) -> dict[str, Any]:
    """Compare ChaosNLI-MNLI items by uid.

    Args:
        mirror: Rows from the mirror (text under ``example``).
        other: Rows from the independent copy (text at top level).

    Returns:
        Counts of shared items and of items whose 100-annotator label
        counts, original labels or text differ.
    """
    m = {r["uid"]: r for r in mirror}
    o = {r["uid"]: r for r in other}
    shared = m.keys() & o.keys()
    counts = [
        u for u in shared if list(m[u]["label_count"]) != list(o[u]["label_count"])
    ]
    old = [u for u in shared if list(m[u]["old_labels"]) != list(o[u]["old_labels"])]
    text = [
        u
        for u in shared
        if any(m[u]["example"][f] != o[u][f] for f in ("premise", "hypothesis"))
    ]
    return {
        "mirror": len(m),
        "independent_copy": len(o),
        "shared": len(shared),
        "label_counts_differ": len(counts),
        "old_labels_differ": len(old),
        "text_differs": len(text),
    }


def mirrors_consistent(result: Mapping[str, Any]) -> bool:
    """Return whether a :func:`check_mirrors` result found no discrepancy.

    Args:
        result: The check result.

    Returns:
        ``True`` if SelfAware matches exactly, the ChaosNLI-MNLI label counts
        match, and every ChaosNLI part has its published size.
    """
    sa, mnli = result["selfaware"], result["chaosnli_mnli"]
    return (
        sa["differing"] == 0
        and sa["only_in_mirror"] == 0
        and sa["only_in_original"] == 0
        and mnli["label_counts_differ"] == 0
        and all(
            v["mirror"] == v["published"]
            for v in result["chaosnli_counts_vs_published"].values()
        )
    )


def check_mirrors(data_dir: Path, fetcher: Fetcher = fetch) -> dict[str, Any]:
    """Compare the SelfAware and ChaosNLI mirrors with independent copies.

    Needs network access (public files only) unless ``fetcher`` serves them.

    Args:
        data_dir: Data directory holding (or receiving) the mirrors.
        fetcher: Downloads the independent copies.

    Returns:
        Per-dataset comparison counts and an overall ``consistent`` flag.
    """
    mirror_sa = loaders.load_selfaware(data_dir).to_dict("records")
    original_sa = json.loads(fetcher(SELFAWARE_GITHUB))["example"]
    mirror_mnli = loaders.load_chaosnli("mnli_m", data_dir).to_dict("records")
    other_mnli = [
        json.loads(x)
        for x in fetcher(TASKSOURCE_MNLI).decode().splitlines()
        if x.strip()
    ]
    res: dict[str, Any] = {
        "selfaware": {
            "source": SELFAWARE_GITHUB,
            **compare_selfaware(mirror_sa, original_sa),
        },
        "chaosnli_mnli": {
            "source": TASKSOURCE_MNLI,
            **compare_chaos_mnli(mirror_mnli, other_mnli),
        },
        "chaosnli_counts_vs_published": {
            s: {"mirror": len(loaders.load_chaosnli(s, data_dir)), "published": n}
            for s, n in PUBLISHED_COUNTS.items()
        },
    }
    res["consistent"] = mirrors_consistent(res)
    logger.info("dataset mirrors consistent: %s", res["consistent"])
    return res
