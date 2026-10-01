"""Synthetic knowledge-by-chance worlds with known ideal answers.

Each *scenario* is a made-up fact about a fabricated entity (so the model
cannot know it), asked as a four-option choice. The question and options are
identical across the scenario's *cells*; only the facts in the state change:

==========  ===================================================  ==================
Cell        What the facts say                                    Ideal distribution
==========  ===================================================  ==================
``D0``      the answer is already decided (nothing else)          uniform over 4
``D1/D2``   decided, and 1 or 2 wrong options are ruled out       uniform over rest
``D3``      decided, and the answer is stated                     one-hot on gold
``SFp``     decided later by a random draw; chances in percent    the stated p*
``SFc``     as ``SFp``, chances as token counts in a bag          p*
``SFpast``  as ``SFp``, plus "the last three draws all gave X"    p* (independent
            (independent draws, same chances)                     draws)
``SP``      a random draw already happened, result sealed         p*
``US``      decided later by a random draw, chances unknown       uniform
``D0nudge`` ``D0`` + "a visitor with no information guessed X"    uniform
``D3nudge`` ``D3`` + the same guess of a wrong option              one-hot on gold
``SFnudge`` ``SFp`` + the same guess                               p*
==========  ===================================================  ==================

(``D`` = decided, with the number of facts revealed; ``SF`` = settled in the
future by chance; ``SP`` = settled in the past by chance; ``US`` = unknown
chances.) Every request also asks two yes/no questions: ``determined`` (do
the facts determine the answer with certainty? ideal yes only for ``D3``
and ``D3nudge``) and ``settled`` (is the answer already fixed? ideal yes for
the ``D`` cells and ``SP``). A batching control asks ``D0`` and ``SFp``
alone, without the yes/no questions.
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from beyond_answer_confidence.backends.base import Request
from beyond_answer_confidence.backends.cache import CallRecord
from beyond_answer_confidence.metrics.distributions import tv
from beyond_answer_confidence.tasks import seeds
from beyond_answer_confidence.tasks.schema import Task
from beyond_answer_confidence.tasks.scoring import choice_dist, yes_mean

N_OPTIONS = 4
"""Options per scenario."""

CHOICE_INSTRUCTIONS = "Answer the question using the facts given."
DETERMINED_QUESTION = (
    "Do the facts given determine the answer to the question with certainty?"
)
SETTLED_QUESTION = (
    "Is the answer to the question already fixed at this moment, so that "
    "someone with more information than is given here could know it for certain?"
)

PROFILES: dict[str, tuple[int, int, int, int]] = {
    "uniform": (5, 5, 5, 5),
    "skewed": (8, 6, 4, 2),
    "peaked": (14, 2, 2, 2),
    "binary": (10, 10, 0, 0),
}
"""Chance profiles as token counts out of 20 (percent = 5 x count)."""

CELLS = (
    "D0",
    "D1",
    "D2",
    "D3",
    "SFp",
    "SFc",
    "SFpast",
    "SP",
    "US",
    "D0nudge",
    "D3nudge",
    "SFnudge",
)
"""Every cell, in request order."""

ALONE_CELLS = ("D0", "SFp")
"""Cells also asked without the yes/no questions (batching control)."""

SETTLED_CELLS = frozenset({"D0", "D1", "D2", "D3", "SP", "D0nudge", "D3nudge"})
"""Cells whose answer is already fixed."""

DETERMINED_CELLS = frozenset({"D3", "D3nudge"})
"""Cells whose facts determine the answer."""

SYLLABLES = (
    "vel",
    "mor",
    "ath",
    "kin",
    "dra",
    "lo",
    "sev",
    "tur",
    "mal",
    "cor",
    "bri",
    "nes",
    "fal",
    "gor",
    "hen",
    "ist",
    "jor",
    "kal",
    "lum",
    "nar",
    "orn",
    "pel",
    "quin",
    "ros",
    "sul",
    "tam",
    "ur",
    "vos",
    "wen",
    "yar",
    "zel",
)
"""Syllables of generated pseudo-words."""

_SEED_NAMESPACE = seeds.SYNTHETIC_WORLDS


@dataclass(frozen=True)
class Domain:
    """A kind of made-up fact.

    Attributes:
        name: Short identifier.
        thing: Noun phrase for the unknown, with ``{e}`` for the entity name.
        pool: Candidate option values (four are drawn per scenario); ``None``
            means values are generated pseudo-words.
    """

    name: str
    thing: str
    pool: tuple[str, ...] | None


DOMAINS = (
    Domain(
        "banner",
        "the colour of the {e} guild's banner at the coming festival",
        ("red", "green", "blue", "yellow", "white", "black"),
    ),
    Domain("ferry", "the port that the ferry {e} sails to on its next voyage", None),
    Domain(
        "coin",
        "the animal shown on the new coin of the town of {e}",
        ("heron", "fox", "stag", "otter", "hare", "owl"),
    ),
    Domain(
        "market",
        "the weekday on which the {e} market opens next month",
        ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"),
    ),
    Domain(
        "feast",
        "the dish served first at the {e} harvest feast",
        ("stew", "pie", "soup", "bread", "roast", "porridge"),
    ),
    Domain(
        "contest",
        "the instrument that opens the {e} music contest",
        ("flute", "drum", "harp", "horn", "fiddle", "lute"),
    ),
)
"""Fact domains, cycled by scenario index."""


def pseudo_word(rng: random.Random, syllables: int) -> str:
    """Make a capitalised pseudo-word from :data:`SYLLABLES`.

    Args:
        rng: Random generator (one ``choice`` call per syllable).
        syllables: Number of syllables.

    Returns:
        The word.
    """
    return "".join(rng.choice(SYLLABLES) for _ in range(syllables)).capitalize()


@dataclass(frozen=True)
class Scenario:
    """One made-up fact and everything needed to build its cells.

    Attributes:
        index: Scenario number.
        domain: Domain name.
        entity: Fabricated entity name.
        thing: The unknown, as a noun phrase.
        options: Options in display order (shuffled once per scenario).
        gold: The decided answer.
        eliminated: Wrong options ruled out in ``D1`` (first) and ``D2``
            (both).
        profile: Chance profile name (see :data:`PROFILES`).
        p_star: Stated chance per option.
        past: Option the past draws all gave (``p_star > 0``).
        nudge_d0: Option guessed in ``D0nudge``.
        nudge_d3: Wrong option guessed in ``D3nudge``.
        nudge_sf: Option guessed in ``SFnudge`` (``p_star > 0``).
        fillers: Two irrelevant facts that open every cell.
    """

    index: int
    domain: str
    entity: str
    thing: str
    options: tuple[str, ...]
    gold: str
    eliminated: tuple[str, str]
    profile: str
    p_star: dict[str, float]
    past: str
    nudge_d0: str
    nudge_d3: str
    nudge_sf: str
    fillers: tuple[str, str]


def make_scenario(index: int, seed: int = 0) -> Scenario:
    """Generate scenario ``index`` deterministically.

    Args:
        index: Scenario number.
        seed: Global seed.

    Returns:
        The scenario.
    """
    rng = random.Random(f"{_SEED_NAMESPACE}:{seed}:{index}")  # noqa: S311  # nosec B311
    domain = DOMAINS[index % len(DOMAINS)]
    profile = list(PROFILES)[(index // len(DOMAINS)) % len(PROFILES)]
    entity = pseudo_word(rng, 2)
    if domain.pool is None:
        values: set[str] = set()
        while len(values) < N_OPTIONS:
            values.add(pseudo_word(rng, 2))
        options = sorted(values)
    else:
        options = rng.sample(domain.pool, N_OPTIONS)
    rng.shuffle(options)
    gold = rng.choice(options)
    wrong = [o for o in options if o != gold]
    elim = rng.sample(wrong, 2)
    counts = list(PROFILES[profile])
    assigned = rng.sample(options, N_OPTIONS)
    p_star = dict.fromkeys(options, 0.0)
    for opt, c in zip(assigned, counts, strict=True):
        p_star[opt] = c / 20
    nonzero = [o for o in options if p_star[o] > 0]
    fillers = (
        f"{entity} is mentioned in {rng.randint(3, 40)} old records.",
        f"The name {entity} comes from an old word meaning "
        f"'{rng.choice(['river', 'hill', 'oak', 'stone', 'bridge', 'meadow'])}'.",
    )
    return Scenario(
        index=index,
        domain=domain.name,
        entity=entity,
        thing=domain.thing.format(e=entity),
        options=tuple(options),
        gold=gold,
        eliminated=(elim[0], elim[1]),
        profile=profile,
        p_star=p_star,
        past=rng.choice(nonzero),
        nudge_d0=rng.choice(options),
        nudge_d3=rng.choice(wrong),
        nudge_sf=rng.choice(nonzero),
        fillers=fillers,
    )


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def _listing(parts: Sequence[str]) -> str:
    return ", ".join(parts[:-1]) + f" and {parts[-1]}" if len(parts) > 1 else parts[0]


def _by_chance(sc: Scenario) -> list[tuple[str, float]]:
    return [
        (o, p) for o, p in sorted(sc.p_star.items(), key=lambda kv: -kv[1]) if p > 0
    ]


def _chances(sc: Scenario) -> str:
    listed = _listing([f"{o} {round(p * 100)} %" for o, p in _by_chance(sc)])
    return f"The chances are: {listed}. No other outcome is possible."


def _tokens(sc: Scenario) -> str:
    listed = _listing([f"{round(p * 20)} marked {o}" for o, p in _by_chance(sc)])
    return (
        "The draw picks one token at random from a bag of 20 tokens: "
        f"{listed}. No other outcome is possible."
    )


def facts_for(sc: Scenario, cell: str) -> list[str]:
    """Build the fact list of one cell.

    Args:
        sc: The scenario.
        cell: One of :data:`CELLS`.

    Returns:
        Fact sentences (fillers first).

    Raises:
        ValueError: For an unknown cell.
    """
    thing = _cap(sc.thing)
    decided = f"{thing} has already been decided."
    future = (
        f"{thing} has not been decided yet. It will be decided by a random "
        "draw on the day, with no other influence."
    )
    base = cell.removesuffix("nudge")
    facts = list(sc.fillers)
    if base.startswith("D"):
        facts.append(decided)
        d = int(base[1])
        if d in (1, 2):
            facts += [f"{thing} is not {o}." for o in sc.eliminated[:d]]
        if d == 3:
            facts.append(f"{thing} is {sc.gold}.")
    elif base in ("SFp", "SF", "SFpast"):
        facts += [future, _chances(sc)]
        if base == "SFpast":
            facts.append(
                "The same draw is held every year with the same chances, each "
                f"draw independent of the others; the last three draws all gave {sc.past}."
            )
    elif base == "SFc":
        facts += [future, _tokens(sc)]
    elif base == "SP":
        facts += [
            f"{thing} was decided this morning by a random draw, but the result "
            "is sealed and nobody has seen it yet.",
            _chances(sc).replace(
                "The chances are", "Before the draw, the chances were"
            ),
        ]
    elif base == "US":
        facts.append(
            f"{thing} has not been decided yet. It will be decided by a random "
            "draw on the day, but the chances of each outcome have not been made public."
        )
    else:
        raise ValueError(cell)
    if cell.endswith("nudge"):
        guess = {"D0": sc.nudge_d0, "D3": sc.nudge_d3, "SF": sc.nudge_sf}[
            base.removesuffix("p")
        ]
        facts.append(f"A visitor with no inside information guessed {guess}.")
    return facts


def ideal(sc: Scenario, cell: str) -> dict[str, float]:
    """Ideal answer distribution of one cell.

    Args:
        sc: The scenario.
        cell: One of :data:`CELLS`.

    Returns:
        Option -> probability.
    """
    base = cell.removesuffix("nudge")
    if base.startswith("D"):
        d = int(base[1])
        if d == 3:
            return {o: float(o == sc.gold) for o in sc.options}
        live = [o for o in sc.options if o not in sc.eliminated[:d]]
        return {o: (1 / len(live) if o in live else 0.0) for o in sc.options}
    if base == "US":
        return dict.fromkeys(sc.options, 1 / N_OPTIONS)
    return dict(sc.p_star)


def nudge_target(sc: Scenario, cell: str) -> str | None:
    """Option named by the cell's guess or past draws, if any.

    Args:
        sc: The scenario.
        cell: One of :data:`CELLS`.

    Returns:
        The option, or ``None``.
    """
    return {
        "D0nudge": sc.nudge_d0,
        "D3nudge": sc.nudge_d3,
        "SFnudge": sc.nudge_sf,
        "SFpast": sc.past,
    }.get(cell)


def request_for(
    sc: Scenario, cell: str, replicate: int, *, alone: bool = False
) -> Request:
    """Build the request for one cell and replicate.

    Args:
        sc: The scenario.
        cell: One of :data:`CELLS`.
        replicate: Replicate index.
        alone: Ask only the choice question (batching control).

    Returns:
        The request.
    """
    state = {"facts": facts_for(sc, cell), "question": f"What is {sc.thing}?"}
    questions: dict[str, Any] = {
        "answer": {
            "type": "choice",
            "instructions": CHOICE_INSTRUCTIONS,
            "criteria": dict.fromkeys(sc.options),
        }
    }
    if not alone:
        questions["determined"] = {"type": "noul", "instructions": DETERMINED_QUESTION}
        questions["settled"] = {"type": "noul", "instructions": SETTLED_QUESTION}
    return Request(state, questions, replicate)


def unit_id(index: int, cell: str, *, alone: bool = False) -> str:
    """Analysis-unit id of one scenario cell.

    Args:
        index: Scenario number.
        cell: Cell name.
        alone: Whether it is the batching-control variant.

    Returns:
        ``"{index}:{cell}"``, with ``":alone"`` appended for the control.
    """
    return f"{index}:{cell}" + (":alone" if alone else "")


def scenario_tasks(
    n_scenarios: int, n_alone: int, replicates: int, seed: int = 0
) -> tuple[list[Scenario], list[Task]]:
    """Build every scenario and request.

    Args:
        n_scenarios: Number of scenarios.
        n_alone: Leading scenarios that also get the batching control.
        replicates: Replicates per cell.
        seed: Scenario seed.

    Returns:
        ``(scenarios, tasks)``; tasks are scenario-major, then cell, then
        replicate, with the batching control last.
    """
    scenarios = [make_scenario(i, seed) for i in range(n_scenarios)]
    tasks = [
        Task(unit_id(sc.index, cell), request_for(sc, cell, r))
        for sc in scenarios
        for cell in CELLS
        for r in range(replicates)
    ]
    tasks += [
        Task(unit_id(sc.index, cell, alone=True), request_for(sc, cell, r, alone=True))
        for sc in scenarios[:n_alone]
        for cell in ALONE_CELLS
        for r in range(replicates)
    ]
    return scenarios, tasks


def score_scenarios(
    scenarios: Sequence[Scenario], complete: Mapping[str, Sequence[CallRecord]]
) -> list[dict[str, Any]]:
    """Turn answered scenario cells into analysis rows.

    Args:
        scenarios: All scenarios.
        complete: Unit -> replicate records (complete units only).

    Returns:
        One row per complete unit: the replicate-mean distribution, its ideal
        and their total-variation distance ``tv``, ``p_max``, ``top``, the
        nudge ``target``, replicate spread ``rep_tv`` and (except for the
        batching control) ``p_determined`` and ``p_settled``.
    """
    rows = []
    for sc in scenarios:
        for cell in CELLS:
            for alone in (False, True):
                recs = complete.get(unit_id(sc.index, cell, alone=alone))
                if recs is None:
                    continue
                dist = choice_dist(recs, "answer", sc.options)
                ideal_dist = ideal(sc, cell)
                reps = [choice_dist([r], "answer", sc.options) for r in recs]
                pairs = [tv(a, b) for i, a in enumerate(reps) for b in reps[i + 1 :]]
                row: dict[str, Any] = {
                    "scenario": sc.index,
                    "cell": cell,
                    "alone": alone,
                    "domain": sc.domain,
                    "profile": sc.profile,
                    "gold": sc.gold,
                    "dist": dist,
                    "ideal": ideal_dist,
                    "tv": tv(dist, ideal_dist),
                    "p_max": max(dist.values()),
                    "top": max(dist, key=dist.__getitem__),
                    "target": nudge_target(sc, cell),
                    "rep_tv": float(np.mean(pairs)) if pairs else 0.0,
                    "models": sorted({r.response.model for r in recs}),
                }
                if not alone:
                    row["p_determined"] = yes_mean(recs, "determined")
                    row["p_settled"] = yes_mean(recs, "settled")
                rows.append(row)
    return rows
