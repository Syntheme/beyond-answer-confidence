"""Synthetic intent data and rows for the intent-experiment unit tests."""

from beyond_answer_confidence.data.intents import Example, IntentDataset

LABELS = [f"i{j}" for j in range(20)]


def tiny_dataset(
    n_labels: int = 4, per_label: int = 12, name: str = "tiny"
) -> IntentDataset:
    """A small synthetic dataset with enough pool examples per intent."""
    labels = tuple(f"intent_{i}" for i in range(n_labels))
    pool = tuple(
        Example(f"{lab} pool {j}", lab) for lab in labels for j in range(per_label)
    )
    dev = tuple(Example(f"{lab} dev {j}", lab) for lab in labels for j in range(4))
    test = tuple(Example(f"{lab} test {j}", lab) for lab in labels for j in range(2))
    oos = tuple(Example(f"out of scope filler {j}", "oos") for j in range(60))
    return IntentDataset(name, labels, pool, dev, test, oos_test=oos[:5], oos_pool=oos)


def without_oos(ds: IntentDataset) -> IntentDataset:
    """The same dataset without out-of-scope splits (Banking77-style)."""
    return IntentDataset(ds.name, ds.labels, ds.pool, ds.dev, ds.test)


def dist(
    gold: str, p_gold: float, top: str | None = None, p_top: float = 0.0
) -> dict[str, float]:
    """Mass ``p_gold`` on gold, optional ``p_top`` on another label, rest even."""
    probs = dict.fromkeys(LABELS, 0.0)
    probs[gold] = p_gold
    if top is not None:
        probs[top] = p_top
    rest = 1.0 - sum(probs.values())
    free = [lab for lab in LABELS if probs[lab] == 0.0]
    for lab in free:
        probs[lab] = rest / len(free)
    return probs


def row(
    item: int,
    cond: str,
    probs: dict[str, float],
    gold: str,
    rep: int = 0,
    arm: str = "main",
) -> dict[str, object]:
    """A knowledge-dial row."""
    nxt = LABELS[(LABELS.index(gold) + 1) % len(LABELS)]
    return {
        "arm": arm,
        "item": item,
        "condition": cond,
        "code_style": "ordinal",
        "example_seed": 0,
        "replicate": rep,
        "K": len(LABELS),
        "gold": gold,
        "swap_target": nxt if cond == "Lswap" else None,
        "first_listed": LABELS[0],
        "probs": probs,
        "model": "m",
    }
