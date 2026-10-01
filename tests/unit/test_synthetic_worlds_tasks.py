import random

import pytest

from beyond_answer_confidence.backends.cache import request_key
from beyond_answer_confidence.tasks import synthetic as syn

# Cache keys (SHA-256 of the request payload, not secrets).
KEY_0_D0 = "27d98307257331d98fb90fef961cf39f3e782b866df6f4b806fa74d99bf7e256"  # pragma: allowlist secret
KEY_1_SFPAST = "8f238112f8f146075e5beecbe4556936e0f8fe73feb813f0d11b783b25a80113"  # pragma: allowlist secret


def test_scenarios_are_deterministic_and_valid() -> None:
    assert syn.make_scenario(5) == syn.make_scenario(5)
    assert syn.make_scenario(5) != syn.make_scenario(5, seed=1)
    for i in range(48):
        sc = syn.make_scenario(i)
        assert len(set(sc.options)) == syn.N_OPTIONS
        assert sc.gold in sc.options
        assert sc.gold not in sc.eliminated
        assert sum(sc.p_star.values()) == pytest.approx(1.0)
        assert sc.p_star[sc.past] > 0
        assert sc.p_star[sc.nudge_sf] > 0
        assert sc.nudge_d3 != sc.gold
        assert sc.domain == syn.DOMAINS[i % len(syn.DOMAINS)].name
        for cell in syn.CELLS:
            assert sum(syn.ideal(sc, cell).values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("index", "cell", "entity", "key"),
    [
        (
            0,
            "D0",
            "Voskin",
            KEY_0_D0,
        ),
        (
            1,
            "SFpast",
            "Falkal",
            KEY_1_SFPAST,
        ),
    ],
)
def test_scenario_requests_are_pinned(
    index: int, cell: str, entity: str, key: str
) -> None:
    # Guards the seed namespace, the draw order and the wording: any change
    # alters every request and orphans existing caches.
    sc = syn.make_scenario(index)
    req = syn.request_for(sc, cell, 0)
    assert sc.entity == entity
    assert request_key("jev-latest", req.state, req.questions, 0) == key


def test_cells_differ_only_in_facts() -> None:
    sc = syn.make_scenario(3)
    reqs = [syn.request_for(sc, c, 0) for c in syn.CELLS]
    assert len({r.state["question"] for r in reqs}) == 1
    assert len({str(r.questions) for r in reqs}) == 1
    assert len({str(r.state["facts"]) for r in reqs}) == len(syn.CELLS)
    alone = syn.request_for(sc, "D0", 0, alone=True)
    assert list(alone.questions) == ["answer"]
    assert list(reqs[0].questions) == ["answer", "determined", "settled"]


def test_facts_reveal_exactly_what_the_cell_says() -> None:
    sc = syn.make_scenario(0)
    d3 = " ".join(syn.facts_for(sc, "D3"))
    assert f"is {sc.gold}." in d3
    assert f"is {sc.gold}." not in " ".join(syn.facts_for(sc, "D0"))
    d2 = " ".join(syn.facts_for(sc, "D2"))
    assert all(f"is not {o}." in d2 for o in sc.eliminated)
    assert "%" in " ".join(syn.facts_for(sc, "SFp"))
    assert "tokens" in " ".join(syn.facts_for(sc, "SFc"))
    assert "sealed" in " ".join(syn.facts_for(sc, "SP"))
    assert "not been made public" in " ".join(syn.facts_for(sc, "US"))
    assert sc.past in syn.facts_for(sc, "SFpast")[-1]
    assert sc.nudge_d3 in syn.facts_for(sc, "D3nudge")[-1]
    assert sc.nudge_sf in syn.facts_for(sc, "SFnudge")[-1]
    assert syn.facts_for(sc, "D0")[:2] == list(sc.fillers)
    with pytest.raises(ValueError, match="XX"):
        syn.facts_for(sc, "XX")


def test_ideal_distributions() -> None:
    sc = syn.make_scenario(2)
    assert syn.ideal(sc, "D3")[sc.gold] == 1.0
    d1 = syn.ideal(sc, "D1")
    assert d1[sc.eliminated[0]] == 0
    assert max(d1.values()) == pytest.approx(1 / 3)
    assert syn.ideal(sc, "D2")[sc.gold] == pytest.approx(0.5)
    assert syn.ideal(sc, "SFnudge") == sc.p_star
    assert set(syn.ideal(sc, "US").values()) == {0.25}
    assert syn.nudge_target(sc, "SFpast") == sc.past
    assert syn.nudge_target(sc, "D0") is None


def test_pseudo_word_uses_syllables() -> None:
    word = syn.pseudo_word(random.Random(1), 3)  # noqa: S311
    assert word[0].isupper()
    rest = word.lower()
    for _ in range(3):
        match = next(s for s in syn.SYLLABLES if rest.startswith(s))
        rest = rest[len(match) :]
    assert rest == ""


def test_scenario_tasks_layout() -> None:
    scenarios, tasks = syn.scenario_tasks(4, 2, 2)
    assert len(scenarios) == 4
    assert len(tasks) == 4 * len(syn.CELLS) * 2 + 2 * len(syn.ALONE_CELLS) * 2
    assert tasks[0].unit == "0:D0"
    assert [t.request.replicate for t in tasks[:2]] == [0, 1]
    assert tasks[-1].unit == "1:SFp:alone"
