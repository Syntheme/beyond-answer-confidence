"""Strict JSON output: non-finite floats become ``null``.

Python's :mod:`json` writes ``NaN`` and ``Infinity`` by default, which are
not JSON and break other parsers. Every output file of this package is
written through :func:`dumps`, which replaces non-finite floats (at any
depth) with ``null`` and then serialises with ``allow_nan=False``.
"""

import json
import math
from typing import Any


def finite(obj: Any) -> Any:
    """Replace non-finite floats with ``None``, recursively.

    Dictionaries and lists are rebuilt (tuples become lists, as JSON would
    write them); every other value is returned unchanged.

    Args:
        obj: A JSON-serialisable object.

    Returns:
        The object without NaN or infinite floats.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [finite(v) for v in obj]
    return obj


def dumps(obj: Any, **kwargs: Any) -> str:
    """Serialise to strict JSON (non-finite floats as ``null``).

    Args:
        obj: A JSON-serialisable object.
        **kwargs: Passed to :func:`json.dumps` (``allow_nan`` is forced off).

    Returns:
        The JSON text.
    """
    kwargs["allow_nan"] = False
    return json.dumps(finite(obj), **kwargs)
