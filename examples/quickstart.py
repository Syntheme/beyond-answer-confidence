"""Quick start: score a few items with the offline fake backend.

Runs without network access or an API key. The fake backend answers every
choice question with half the probability on the first option, so the
numbers only show the shape of the output.

    uv run python examples/quickstart.py
"""

import json
import tempfile
from pathlib import Path

from beyond_answer_confidence.backends.fake import FakeBackend
from beyond_answer_confidence.runner import run_items
from beyond_answer_confidence.tasks.schema import Item

ITEMS = [
    Item(
        unit=f"toy:{i}",
        state={"case": f"Toy case {i}: which colour is listed as the answer?"},
        options=("A", "B", "C", "D"),
        gold="ABCD"[i % 4],
        instructions="Pick the option that answers the case.",
        descriptions={"A": "red", "B": "green", "C": "blue", "D": "yellow"},
        nouls={"known": "Is the answer stated in the case?"},
        info={"set": "toy", "index": i},
    )
    for i in range(8)
]


def main() -> None:
    """Collect answers through a temporary cache and print a summary."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "cache" / "quickstart.jsonl"
        rows, bookkeeping = run_items(
            ITEMS,
            cache=cache,
            model="fake-model",
            replicates=2,
            backend=FakeBackend,
            max_input_tokens=100_000,
            label="quickstart",
        )
        # A second pass is answered entirely from the cache.
        _, again = run_items(
            ITEMS, cache=cache, model="fake-model", replicates=2, backend=None
        )
    summary = {
        "items": len(rows),
        "accuracy": sum(bool(r["correct"]) for r in rows) / len(rows),
        "mean_p_max": sum(r["p_max"] for r in rows) / len(rows),
        "mean_p_known": sum(r["p_known"] for r in rows) / len(rows),
        "first_pass_api_calls": bookkeeping["new_api_calls"],
        "second_pass_api_calls": again["new_api_calls"],
        "first_row": rows[0],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
