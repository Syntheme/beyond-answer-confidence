"""Tables and figures drawn from experiment outputs."""

from pathlib import Path

from beyond_answer_confidence.settings import Settings


def reportable() -> list[str]:
    """Return the names that have a table or figure reporter.

    Returns:
        Sorted names.
    """
    from beyond_answer_confidence.reporting.figures import FIGURES
    from beyond_answer_confidence.reporting.tables import TABLES

    return sorted(set(FIGURES) | set(TABLES))


def report(name: str, settings: Settings, *, figures: bool = True) -> list[Path]:
    """Write the tables (and figures) of one experiment or analysis.

    Args:
        name: Experiment or analysis name.
        settings: Settings (locate the outputs).
        figures: Also draw figures.

    Returns:
        The files written.

    Raises:
        KeyError: If no reporter exists for ``name``.
    """
    from beyond_answer_confidence.reporting.figures import FIGURES
    from beyond_answer_confidence.reporting.tables import TABLES

    if name not in FIGURES and name not in TABLES:
        raise KeyError(f"no reporter for {name!r}")
    written: list[Path] = []
    if name in TABLES:
        written += TABLES[name](settings)
    if figures and name in FIGURES:
        written += FIGURES[name](settings)
    return written
