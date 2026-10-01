"""Read experiment rows into data frames for the offline analyses.

Rows are parsed with :func:`pandas.read_json` and its default (fast, not
round-trip exact) float parser. Some statistics depend on exact ties between
parsed values (for example thresholds matched to a coverage, where tied
scores share the boundary), so the parser is kept fixed for reproducibility
of released results. Pass
``precise_float=True`` for round-trip exact values, or use
:func:`records_frame`, which parses with :mod:`json`.
"""

from pathlib import Path

import pandas as pd

from beyond_answer_confidence.experiments.base import read_jsonl
from beyond_answer_confidence.settings import Settings


def read_frame(
    path: Path,
    *,
    dtype: dict[str, type] | None = None,
    precise_float: bool = False,
) -> pd.DataFrame:
    """Read a JSON-lines file of rows.

    Args:
        path: ``rows.jsonl`` file.
        dtype: Column types to force (e.g. ``{"month": str}``); other
            columns are inferred.
        precise_float: Use the round-trip exact float parser.

    Returns:
        One row per line.
    """
    if dtype is None:
        return pd.read_json(path, lines=True, precise_float=precise_float)
    return pd.read_json(path, lines=True, dtype=dtype, precise_float=precise_float)


def experiment_frame(
    settings: Settings,
    experiment: str,
    *,
    dtype: dict[str, type] | None = None,
    precise_float: bool = False,
) -> pd.DataFrame:
    """Read ``<output_dir>/<experiment>/rows.jsonl``.

    Args:
        settings: Run settings (locate the outputs).
        experiment: Experiment name.
        dtype: Column types to force; other columns are inferred.
        precise_float: Use the round-trip exact float parser.

    Returns:
        The experiment's rows.
    """
    return read_frame(
        settings.out(experiment) / "rows.jsonl",
        dtype=dtype,
        precise_float=precise_float,
    )


def records_frame(settings: Settings, experiment: str) -> pd.DataFrame:
    """Read an experiment's rows with the exact JSON parser, one dict per row.

    Args:
        settings: Run settings (locate the outputs).
        experiment: Experiment name.

    Returns:
        ``pandas.DataFrame`` of the parsed rows (columns inferred from the
        dicts, so missing keys become NaN/None).
    """
    return pd.DataFrame(read_jsonl(settings.out(experiment) / "rows.jsonl"))
