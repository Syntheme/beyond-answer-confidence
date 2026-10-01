"""Items and tasks: the unit of work every experiment builds.

An :class:`Item` is one case: a state, an optional choice question and any
number of yes/no and score questions. Each item expands into one
:class:`Task` per replicate; tasks sharing a ``unit`` are averaged when
scored.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from beyond_answer_confidence.backends.base import Request
from beyond_answer_confidence.backends.cache import request_key


@dataclass(frozen=True)
class Task:
    """One request belonging to one analysis unit.

    Attributes:
        unit: Identifier of the analysis unit; replicates of a unit share it.
        request: The request (its ``replicate`` field separates repeats).
    """

    unit: str
    request: Request


@dataclass(frozen=True)
class Item:
    """A generic item: an optional choice question plus yes/no and score questions.

    Attributes:
        unit: Unique analysis-unit id.
        state: Request state.
        options: Choice option keys in display order (empty = no choice).
        gold: Correct option key, if any.
        instructions: Choice instructions.
        descriptions: Option key -> description (``None`` = bare keys).
        nouls: Yes/no question name -> question text.
        info: Metadata copied into the analysis row.
        scores: Score name -> ``(instructions, levels)``; levels are the
            rubric descriptions for scores 0, 1, 2, ...
    """

    unit: str
    state: Mapping[str, Any]
    options: tuple[str, ...] = ()
    gold: str | None = None
    instructions: str = ""
    descriptions: Mapping[str, Any] | None = None
    nouls: Mapping[str, str] = field(default_factory=dict)
    info: Mapping[str, Any] = field(default_factory=dict)
    scores: Mapping[str, tuple[str, tuple[str, ...]]] = field(default_factory=dict)

    def request(self, replicate: int) -> Request:
        """Build the request for one replicate.

        The question order (choice, then yes/no, then scores) is part of the
        wire payload; keep it stable.

        Args:
            replicate: Replicate index.

        Returns:
            The request.
        """
        questions: dict[str, Any] = {}
        if self.options:
            criteria = (
                {o: self.descriptions[o] for o in self.options}
                if self.descriptions is not None
                else dict.fromkeys(self.options)
            )
            questions["answer"] = {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": criteria,
            }
        for name, text in self.nouls.items():
            questions[name] = {"type": "noul", "instructions": text}
        for name, (text, levels) in self.scores.items():
            questions[name] = {
                "type": "score",
                "instructions": text,
                "criteria": list(levels),
            }
        return Request(dict(self.state), questions, replicate)


def item_tasks(items: Sequence[Item], replicates: int) -> list[Task]:
    """Expand items into tasks (replicate-major, so early chunks cover all items).

    Args:
        items: Items.
        replicates: Replicates per item.

    Returns:
        Tasks.
    """
    return [Task(it.unit, it.request(r)) for r in range(replicates) for it in items]


def order_collisions(tasks: Sequence[Task], model: str) -> int:
    """Count tasks that share a cache key but would send a different payload.

    The cache key hashes the request with sorted keys, while backends send
    dictionaries in insertion order, so two requests that differ only in
    option order map to one cache entry. Experiments that vary option order
    must give each order its own ``replicate`` index.

    Args:
        tasks: Tasks.
        model: Model name.

    Returns:
        Number of colliding tasks (0 is required).
    """
    seen: dict[str, str] = {}
    bad = 0
    for t in tasks:
        r = t.request
        key = request_key(model, r.state, r.questions, r.replicate)
        sent = json.dumps({"s": r.state, "q": r.questions}, ensure_ascii=False)
        if seen.setdefault(key, sent) != sent:
            bad += 1
    return bad
