import hashlib
import json
from collections import Counter
from typing import Any

import pytest
from intent_helpers import tiny_dataset, without_oos

from beyond_answer_confidence.backends.base import (
    ChoiceAnswer,
    Response,
    parse_response,
)
from beyond_answer_confidence.backends.cache import CallRecord, request_key
from beyond_answer_confidence.backends.fake import fake_body
from beyond_answer_confidence.data.intents import Example
from beyond_answer_confidence.tasks import out_of_scope as oos
from beyond_answer_confidence.tasks.intents import (
    EXAMPLE_COUNTS,
    EXAMPLE_DEPENDENT,
    LETTER_CONDITIONS,
    MAX_EXAMPLES,
    ChoiceSpec,
    CodeStyle,
    Condition,
    Planner,
    assign_codes,
    build_choice,
    dial_row,
    label_probs,
    plan_cells,
    sample_examples,
    state_for,
)

DS = tiny_dataset()
EXAMPLES = sample_examples(DS, seed=0)
FILLER = [ex.text for ex in DS.oos_pool]
ITEM = DS.test[0]
MODEL = "jev-latest"


def build(condition: Condition, perm_seed: int = 0, key: str = "item0") -> ChoiceSpec:
    return build_choice(ITEM, key, DS, condition, EXAMPLES, FILLER, perm_seed)


def examples_of(spec: ChoiceSpec, code: str, field: str = "examples") -> Any:
    desc = spec.criteria[code]
    assert isinstance(desc, dict)
    return desc[field]


def key_of(req: Any) -> str:
    return request_key(MODEL, req.state, req.questions, req.replicate)


# --- byte compatibility: fixed expected keys ---------------------------------------


def test_knowledge_dial_requests_are_byte_compatible() -> None:
    ds = tiny_dataset(n_labels=5, per_label=12, name="banking77")
    planner = Planner(ds, "test", [ex.text for ex in ds.oos_pool])
    cells = plan_cells(
        planner.examples, per_intent=1, subset_per_intent=1, replicates=2
    )
    plan = [planner.planned(c) for c in cells]
    keys = [key_of(p.request) for p in plan]
    assert len(keys) == 190
    digest = hashlib.sha256("".join(keys).encode()).hexdigest()
    expected = "a190cd5c5ea34ebe7b3d965f9b3ff22e81dcf3de24ca98e6fa3355c649dad892"  # pragma: allowlist secret
    assert digest == expected
    irrel = next(p for p in plan if p.cell.condition is Condition.IRREL)
    assert (
        key_of(irrel.request)
        == (
            "27625ee65cd590ad3ffb208a5b1b5c4ea2b16f5ae7f4417dd3f7b6671ab55cf4"  # pragma: allowlist secret
        )
    )
    letters = next(p for p in plan if p.cell.arm == "letters")
    assert (
        key_of(letters.request)
        == (
            "67477bcb0af174934cd34502476563e8190d0fd1a41f59e7a6c54a4348ee4f8d"  # pragma: allowlist secret
        )
    )


@pytest.mark.parametrize(
    ("n_labels", "name", "has_oos", "per_intent", "rep", "n", "digest"),
    [
        (6, "clinc150", True, 1, 1, 11,
         "08afb6f9bde04ec6fc4b59a4a1e6f5b8e9ac3480a0f693e0320563492ee0771c"),  # pragma: allowlist secret
        (20, "banking77", False, 2, 0, 40,
         "b9e9492a9c632128b1a31d1d26580057f15234194b7b97c94443498b82cac182"),  # pragma: allowlist secret
    ],
)  # fmt: skip
def test_out_of_scope_requests_are_byte_compatible(
    n_labels: int,
    name: str,
    has_oos: bool,
    per_intent: int,
    rep: int,
    n: int,
    digest: str,
) -> None:
    ds = tiny_dataset(n_labels=n_labels, name=name)
    if not has_oos:
        ds = without_oos(ds)
    schema, items = oos.build_items(ds, per_intent=per_intent)
    keys = [
        key_of(oos.scope_request(it, oos.questions_for(it, schema), rep))
        for it in items
    ]
    assert len(keys) == n
    assert hashlib.sha256("".join(keys).encode()).hexdigest() == digest


def test_out_of_scope_keys_round_trip_through_seed_form() -> None:
    key = oos.item_key("banking77", True, 12)
    assert key == "banking77:oos:12"
    seeded = oos.seed_key(key)
    assert seeded != key
    assert seeded.startswith("banking77:")
    assert seeded.endswith(":oos:12")
    assert oos.normalise_key(seeded) == key
    assert oos.normalise_key(key) == key
    assert oos.normalise_key("test:3") == "test:3"
    with pytest.raises(ValueError, match="item key"):
        oos.seed_key("no-colon")


# --- conditions -------------------------------------------------------------------


def test_codes_are_opaque_and_cover_all_labels() -> None:
    codes = assign_codes(DS.labels, "x", 0)
    assert sorted(codes.values()) == sorted(DS.labels)
    assert all(c.startswith("INTENT_") for c in codes)
    assert not any(lab in code for code in codes for lab in DS.labels)


def test_code_assignment_is_reproducible_and_varies_with_seed() -> None:
    assert assign_codes(DS.labels, "x", 0) == assign_codes(DS.labels, "x", 0)
    assignments = {tuple(assign_codes(DS.labels, "x", s).values()) for s in range(10)}
    assert len(assignments) > 1


def test_examples_are_nested_across_doses() -> None:
    specs = {
        c: build(c) for c in (Condition.L1, Condition.L2, Condition.L4, Condition.L8)
    }
    code = specs[Condition.L8].gold_code
    ex8 = examples_of(specs[Condition.L8], code)
    for cond, spec in specs.items():
        assert examples_of(spec, code) == ex8[: EXAMPLE_COUNTS[cond]]


def test_code_mapping_is_identical_across_conditions() -> None:
    assert len({tuple(build(c).code_to_label.items()) for c in Condition}) == 1


def test_l0_gives_no_information() -> None:
    spec = build(Condition.L0)
    assert all(v is None for v in spec.criteria.values())
    assert spec.code_to_label[spec.gold_code] == ITEM.label


def test_examples_belong_to_their_intent() -> None:
    spec = build(Condition.L8)
    for code in spec.criteria:
        label = spec.code_to_label[code]
        assert all(text.startswith(label) for text in examples_of(spec, code))


def test_name_conditions() -> None:
    spec = build(Condition.NAME)
    assert all(
        spec.criteria[c] == {"name": lab} for c, lab in spec.code_to_label.items()
    )
    spec8 = build(Condition.NAME8)
    assert len(examples_of(spec8, spec8.gold_code)) == 8


def test_irrelevant_filler_matches_example_length() -> None:
    spec = build(Condition.IRREL)
    for code in spec.criteria:
        notes = examples_of(spec, code, "notes")
        target = sum(len(t) for t in EXAMPLES[spec.code_to_label[code]][:8])
        total = sum(len(t) for t in notes)
        assert total >= target
        assert total - len(notes[-1]) < target  # stops once the target is reached
        assert set(notes) <= set(FILLER)
        assert len(set(notes)) == len(notes)


def test_irrelevant_needs_enough_filler() -> None:
    with pytest.raises(ValueError, match="filler"):
        build_choice(ITEM, "k", DS, Condition.IRREL, EXAMPLES, FILLER[:2])


def test_swap_is_a_derangement_and_tracks_gold_examples() -> None:
    spec = build(Condition.SWAP)
    assert spec.swap_code is not None
    assert spec.swap_code != spec.gold_code
    for code in spec.criteria:
        shown = examples_of(spec, code)[0].split(" pool ")[0]
        assert shown != spec.code_to_label[code]
    assert all(t.startswith(ITEM.label) for t in examples_of(spec, spec.swap_code))


def test_unknown_gold_label_rejected() -> None:
    with pytest.raises(ValueError, match="not in label set"):
        build_choice(Example("x", "nope"), "k", DS, Condition.L0, EXAMPLES, FILLER)


def test_sample_examples_requires_enough_pool() -> None:
    with pytest.raises(ValueError, match="pool examples"):
        sample_examples(tiny_dataset(per_label=MAX_EXAMPLES - 1), seed=0)


def test_question_and_state_shape() -> None:
    q = build(Condition.L0).question()
    assert q["type"] == "choice"
    assert "criteria" in q
    assert state_for("hi") == {"customer_message": "hi"}


def test_letter_codes_have_no_digits_or_vowels_and_keep_order() -> None:
    big = tuple(f"intent_{i}" for i in range(150))
    ordinal = assign_codes(big, "x", 0)
    letters = assign_codes(big, "x", 0, CodeStyle.LETTERS)
    assert len(set(letters)) == 150
    assert not any(ch.isdigit() for code in letters for ch in code)
    assert not any(ch in "AEIOU" for code in letters for ch in code[7:])
    assert list(letters.values()) == list(ordinal.values())


def test_label_probs_maps_codes_back_and_normalises() -> None:
    spec = build(Condition.L0)
    codes = list(spec.code_to_label)
    probs = {c: (0.7 if i == 0 else 0.1) for i, c in enumerate(codes)}
    mapped = label_probs(spec, probs)
    assert mapped[spec.code_to_label[codes[0]]] == pytest.approx(0.7)
    assert sum(mapped.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="do not match"):
        label_probs(spec, {"INTENT_X": 1.0})


# --- knowledge-dial plan ----------------------------------------------------------


def test_plan_arms_and_counts() -> None:
    ds = tiny_dataset(n_labels=5, name="banking77")
    planner = Planner(ds, "dev", FILLER)
    cells = plan_cells(
        planner.examples, per_intent=2, subset_per_intent=1, replicates=3
    )
    plan = [planner.planned(c) for c in cells]
    arms = Counter(p.cell.arm for p in plan)
    assert arms["main"] == 5 * 2 * len(Condition) * 3
    assert arms["seeds"] == 5 * 1 * 2 * len(EXAMPLE_DEPENDENT)
    assert arms["letters"] == 5 * 1 * len(LETTER_CONDITIONS) * 3
    letters = [p for p in plan if p.cell.arm == "letters"]
    assert all(not any(ch.isdigit() for ch in p.spec.gold_code) for p in letters)
    first = plan[0].cell.item
    main = [
        p
        for p in plan
        if p.cell.arm == "main"
        and p.cell.item == first
        and p.cell.condition is Condition.L0
    ]
    assert len({json.dumps(p.request.questions, sort_keys=True) for p in main}) == 1
    assert {p.request.replicate for p in main} == {0, 1, 2}


def test_dial_row_maps_probabilities_back_to_intents() -> None:
    ds = tiny_dataset(n_labels=5, name="banking77")
    planner = Planner(ds, "dev", FILLER)
    cells = plan_cells(
        planner.examples, per_intent=1, subset_per_intent=1, replicates=1
    )
    for cell in cells:
        p = planner.planned(cell)
        body = fake_body(p.request.questions, MODEL)
        rec = CallRecord("k", parse_response(body), True, None, None)
        row = dial_row(p, rec)
        assert set(row["probs"]) == set(ds.labels)
        assert row["gold"] == ds.dev[cell.item].label
        assert row["choice"] == row["first_listed"]  # the fake favours the first code
        assert row["probs"][row["gold"]] == pytest.approx(row["p_gold"])
        if cell.condition is Condition.SWAP:
            assert row["swap_target"] not in (None, row["gold"])
        json.dumps(row)


# --- out of scope -------------------------------------------------------------


def test_heldout_is_seeded_and_sized() -> None:
    labels = [f"i{k}" for k in range(77)]
    held = oos.heldout_intents(labels)
    assert len(held) == 10
    assert held == oos.heldout_intents(labels)
    assert set(held) <= set(labels)


def test_build_items_holds_out_intents_without_oos_split() -> None:
    ds = without_oos(tiny_dataset(n_labels=20, name="banking77"))
    schema, items = oos.build_items(ds, per_intent=2)
    held = set(oos.heldout_intents(ds.labels))
    assert not held & set(schema)
    assert all(it.gold not in held for it in items if not it.oos)
    assert sum(it.oos for it in items) == sum(ex.label in held for ex in ds.test)


def test_build_items_uses_oos_split() -> None:
    ds = tiny_dataset(n_labels=6, name="clinc150")
    schema, items = oos.build_items(ds, per_intent=1)
    assert schema == list(ds.labels)
    assert sum(it.oos for it in items) == len(ds.oos_test)


def test_questions_have_three_readouts() -> None:
    q = oos.questions_for(
        oos.ScopeItem(oos.item_key("x", False, 0), "hello", False, "a"), ["a", "b", "c"]
    )
    assert set(q.questions) == {"closed", "with_oos", "fits"}
    assert sorted(q.closed.values()) == ["a", "b", "c"]
    assert oos.OOS_OPTION in q.with_oos.values()
    assert q.questions["fits"]["type"] == "noul"


def _record(q: oos.ScopeQuestions, p_oos: float, fits: float) -> CallRecord:
    oos_code = next(c for c, lab in q.with_oos.items() if lab == oos.OOS_OPTION)
    other = next(c for c in q.with_oos if c != oos_code)
    first = next(iter(q.closed))
    probs_b = dict.fromkeys(q.with_oos, 0.0) | {oos_code: p_oos, other: 1 - p_oos}
    probs_a = dict.fromkeys(q.closed, 0.0) | {first: 1.0}
    body = {
        "model": "m",
        "answers": {
            "closed": {"type": "choice", "choice": first, "probabilities": probs_a},
            "with_oos": {
                "type": "choice",
                "choice": oos_code,
                "probabilities": probs_b,
            },
            "fits": {"type": "noul", "noul": fits},
        },
    }
    return CallRecord("k", parse_response(body), False, None, None)


def test_score_item_averages_replicates() -> None:
    q = oos.questions_for(
        oos.ScopeItem(oos.item_key("x", True, 0), "hello", True, None), ["a", "b", "c"]
    )
    s = oos.score_item([_record(q, 0.6, 0.2), _record(q, 0.8, 0.4)], q)
    assert s["p_oos"] == pytest.approx(0.7)
    assert s["p_fits"] == pytest.approx(0.3)
    assert s["with_oos_choice"] == oos.OOS_OPTION
    assert s["closed_p_max"] == pytest.approx(1.0)


def test_confidence_is_parsed_when_reported() -> None:
    body = {
        "model": "m",
        "answers": {
            "q": {"type": "choice", "choice": "a", "confidence": 0.4,
                  "probabilities": {"a": 0.7, "b": 0.3}},
        },
    }  # fmt: skip
    resp: Response = parse_response(body)
    assert resp.choices["q"] == ChoiceAnswer({"a": 0.7, "b": 0.3}, "a", 0.4)
