import subprocess
import sys

import pytest

from beyond_answer_confidence.experiments import registry
from beyond_answer_confidence.experiments.base import Analysis, Experiment
from beyond_answer_confidence.experiments.registry import Entry


def test_names_are_unique_and_ordered() -> None:
    names = registry.names()
    assert len(names) == len(set(names))
    assert names[0] == "instrument_checks"
    assert "knowledge_dial" in names
    assert "gliner_comparison" in names


def test_every_entry_loads_with_matching_name() -> None:
    objs = registry.load_all()
    assert [o.name for o in objs] == registry.names()
    assert all(isinstance(o, Experiment | Analysis) for o in objs)
    assert all(o.summary for o in objs)


def test_experiment_names_are_experiments() -> None:
    names = registry.experiment_names()
    assert "synthetic_worlds" in names
    assert "error_detection" not in names
    assert "deep_ensemble" not in names


def test_duplicate_names_raise() -> None:
    entries = (Entry("a", "m", "X"), Entry("b", "m", "Y"), Entry("a", "n", "Z"))
    with pytest.raises(ValueError, match="duplicate registry name"):
        registry.check_unique(entries)


def test_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="unknown experiment"):
        registry.load("no_such_thing")


def test_load_entry_checks_type_and_name() -> None:
    with pytest.raises(TypeError, match="not an Experiment"):
        registry.load_entry(Entry("toy", "toy_experiment", "NOT_REGISTRABLE"))
    with pytest.raises(ValueError, match="registered as"):
        registry.load_entry(Entry("other", "toy_experiment", "EXPERIMENT"))
    assert registry.load_entry(Entry("toy_analysis", "toy_experiment", "ANALYSIS"))


def test_cache_names(monkeypatch: pytest.MonkeyPatch) -> None:
    assert registry.cache_name("knowledge_dial") == "intents"
    assert registry.cache_name("synthetic_worlds") == "synthetic_worlds"
    monkeypatch.setattr(
        registry,
        "ENTRIES",
        (Entry("toy_experiment", "toy_experiment", "EXPERIMENT"),),
    )
    assert registry.cache_name("toy_experiment") == "toy_cache"


def test_importing_the_registry_and_cli_imports_no_experiment() -> None:
    code = (
        "import sys, beyond_answer_confidence.cli, beyond_answer_confidence.experiments.registry;"
        "bad = [m for m in sys.modules if m.startswith("
        "('beyond_answer_confidence.experiments.knowledge', 'beyond_answer_confidence.analyses',"
        " 'torch', 'transformers', 'matplotlib'))];"
        "print(bad)"
    )
    out = subprocess.run(  # noqa: S603 - fixed interpreter and code
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
