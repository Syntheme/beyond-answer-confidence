"""The ordered list of every experiment and offline analysis.

Each entry names a module and the attribute holding its
:class:`~beyond_answer_confidence.experiments.base.Experiment` or
:class:`~beyond_answer_confidence.experiments.base.Analysis`. Modules are imported only
when an entry is used, so starting the command line never imports heavy
optional dependencies, and adding a module is one more line in
:data:`ENTRIES`.
"""

import importlib
from collections.abc import Iterable
from dataclasses import dataclass

from beyond_answer_confidence.experiments.base import Analysis, Experiment

Registered = Experiment | Analysis
"""Anything the registry can hold."""


@dataclass(frozen=True)
class Entry:
    """One registered object.

    Attributes:
        name: Command-line name (must equal the object's ``name``).
        module: Dotted module path.
        attribute: Module attribute holding the object.
    """

    name: str
    module: str
    attribute: str


_EXP = "beyond_answer_confidence.experiments"
_ANA = "beyond_answer_confidence.analyses"

ENTRIES: tuple[Entry, ...] = (
    # Intent classification (shared ``intents`` cache).
    Entry("instrument_checks", f"{_EXP}.instrument_checks", "EXPERIMENT"),
    Entry("sampling_frequencies", f"{_EXP}.sampling_frequencies", "EXPERIMENT"),
    Entry("knowledge_dial", f"{_EXP}.knowledge_dial", "EXPERIMENT"),
    Entry("out_of_scope", f"{_EXP}.out_of_scope", "EXPERIMENT"),
    # Local comparators and analyses of the intent experiments.
    Entry("deep_ensemble", f"{_EXP}.comparators.deep_ensemble", "TRAINING"),
    Entry("deep_ensemble_comparison", f"{_EXP}.comparators.deep_ensemble", "ANALYSIS"),
    Entry("gliner", f"{_EXP}.comparators.gliner", "INFERENCE"),
    Entry("gliner_comparison", f"{_EXP}.comparators.gliner", "ANALYSIS"),
    Entry("error_detection", f"{_ANA}.error_detection", "ANALYSIS"),
    # Item-set experiments.
    Entry("synthetic_worlds", f"{_EXP}.synthetic_worlds", "EXPERIMENT"),
    Entry("knowledge_boundary", f"{_EXP}.knowledge_boundary", "EXPERIMENT"),
    Entry("evidence_sufficiency", f"{_EXP}.evidence_sufficiency", "EXPERIMENT"),
    Entry("benchmarks", f"{_EXP}.benchmarks", "EXPERIMENT"),
    Entry("follow_up_questions", f"{_EXP}.follow_up_questions", "EXPERIMENT"),
    Entry("second_look", f"{_EXP}.second_look", "EXPERIMENT"),
    Entry("contrastive_facts", f"{_EXP}.contrastive_facts", "EXPERIMENT"),
    Entry("option_count", f"{_EXP}.option_count", "EXPERIMENT"),
    Entry("stated_odds", f"{_EXP}.stated_odds", "EXPERIMENT"),
    Entry("open_benchmarks", f"{_EXP}.open_benchmarks", "EXPERIMENT"),
    Entry("inferred_settledness", f"{_EXP}.inferred_settledness", "EXPERIMENT"),
    Entry("shortcut_controls", f"{_EXP}.shortcut_controls", "EXPERIMENT"),
    # Offline analyses over the rows above.
    Entry("cutoff_calibration", f"{_ANA}.cutoff_calibration", "ANALYSIS"),
    Entry("knowledge_cutoff", f"{_ANA}.knowledge_cutoff", "ANALYSIS"),
    Entry("recalibration", f"{_ANA}.recalibration", "ANALYSIS"),
    Entry("combined_signals", f"{_ANA}.combined_signals", "ANALYSIS"),
    Entry("abstention", f"{_ANA}.abstention", "ANALYSIS"),
    Entry("popularity_and_dose", f"{_ANA}.popularity_and_dose", "ANALYSIS"),
    Entry("replicate_averaging", f"{_ANA}.replicate_averaging", "ANALYSIS"),
    Entry("smece_coverage", f"{_ANA}.smece_coverage", "ANALYSIS"),
    Entry("shortcut_baselines", f"{_ANA}.shortcut_baselines", "ANALYSIS"),
    # New experiments and analyses: append one Entry per object.
)
"""Every registered object, in listing order."""


def check_unique(entries: Iterable[Entry]) -> None:
    """Refuse duplicate names.

    Args:
        entries: Entries to check.

    Raises:
        ValueError: If two entries share a name.
    """
    seen: set[str] = set()
    for e in entries:
        if e.name in seen:
            raise ValueError(f"duplicate registry name: {e.name!r}")
        seen.add(e.name)


check_unique(ENTRIES)


def names() -> list[str]:
    """Return every registered name, in order (imports nothing).

    Returns:
        The names.
    """
    return [e.name for e in ENTRIES]


def entry(name: str) -> Entry:
    """Look up an entry by name.

    Args:
        name: Registered name.

    Returns:
        The entry.

    Raises:
        KeyError: If the name is not registered.
    """
    for e in ENTRIES:
        if e.name == name:
            return e
    raise KeyError(f"unknown experiment or analysis {name!r}; see `list`")


def load_entry(e: Entry) -> Registered:
    """Import an entry's module and return its object.

    Args:
        e: The entry.

    Returns:
        The experiment or analysis.

    Raises:
        TypeError: If the attribute is neither an experiment nor an analysis.
        ValueError: If the object's name differs from the entry's.
    """
    obj = getattr(importlib.import_module(e.module), e.attribute)
    if not isinstance(obj, Experiment | Analysis):
        raise TypeError(f"{e.module}.{e.attribute} is not an Experiment or Analysis")
    if obj.name != e.name:
        raise ValueError(
            f"{e.module}.{e.attribute} is named {obj.name!r}, registered as {e.name!r}"
        )
    return obj


def load(name: str) -> Registered:
    """Import and return one registered object.

    Args:
        name: Registered name.

    Returns:
        The experiment or analysis.
    """
    return load_entry(entry(name))


def load_all() -> list[Registered]:
    """Import every registered object, in order.

    Returns:
        The experiments and analyses.
    """
    return [load_entry(e) for e in ENTRIES]


def experiment_names() -> list[str]:
    """Return the names of the registered experiments (imports every module).

    Returns:
        Names of :class:`Experiment` entries, in order.
    """
    return [o.name for o in load_all() if isinstance(o, Experiment)]


def cache_name(name: str) -> str:
    """Return the request cache an experiment uses.

    Experiments that share a cache declare it in a module-level ``CACHE``
    constant; the others use their own name.

    Args:
        name: Registered experiment name.

    Returns:
        Cache name (``<data_dir>/cache/<cache name>.jsonl``).
    """
    e = entry(name)
    return str(getattr(importlib.import_module(e.module), "CACHE", name))
