"""Load experiment configurations from TOML into typed dataclasses.

Every experiment defines a frozen dataclass whose defaults are its standard
sizes and seeds. A TOML file (``configs/<experiment>.toml``) and
``--set key=value`` overrides from the command line are applied on top.
"""

import dataclasses
import tomllib
import types
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _coerce(value: Any, annotation: Any) -> Any:
    """Convert a TOML or command-line value to a field's annotated type.

    Args:
        value: Raw value (strings come from the command line).
        annotation: The field's type annotation.

    Returns:
        The converted value.

    Raises:
        ValueError: If the value cannot be converted.
    """
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if value is None or (isinstance(value, str) and value.lower() == "none"):
            return None
        return _coerce(value, args[0])
    if origin is tuple:
        items = value.split(",") if isinstance(value, str) else list(value)
        targs = typing.get_args(annotation)
        inner = targs[0] if targs else str
        return tuple(_coerce(v, inner) for v in items)
    if annotation is bool:
        if isinstance(value, str):
            if value.lower() in {"1", "true", "yes"}:
                return True
            if value.lower() in {"0", "false", "no"}:
                return False
            raise ValueError(f"not a boolean: {value!r}")
        return bool(value)
    if annotation in (int, float, str):
        return annotation(value)
    return value


def load_config[C](
    config_type: type[C],
    path: Path | None = None,
    overrides: Sequence[str] = (),
) -> C:
    """Build a configuration from defaults, an optional TOML file and overrides.

    Args:
        config_type: A dataclass type.
        path: Optional TOML file with a flat table of field values.
        overrides: ``key=value`` strings; they take precedence over the file.

    Returns:
        The configuration.

    Raises:
        ValueError: On unknown keys or malformed overrides.
    """
    if not dataclasses.is_dataclass(config_type):
        raise TypeError(f"{config_type!r} is not a dataclass")
    hints = typing.get_type_hints(config_type)
    names = {f.name for f in dataclasses.fields(config_type)}
    values: dict[str, Any] = {}
    if path is not None:
        with path.open("rb") as f:
            values |= tomllib.load(f)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not key=value")
        values[key.strip()] = raw.strip()
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown {config_type.__name__} keys: {sorted(unknown)}")
    kwargs = {k: _coerce(v, hints[k]) for k, v in values.items()}
    return config_type(**kwargs)


def as_dict(config: Any) -> dict[str, Any]:
    """Return a configuration as a JSON-friendly dict (for run summaries).

    Args:
        config: A dataclass instance.

    Returns:
        Field -> value, with tuples as lists.
    """

    def plain(v: Any) -> Any:
        if isinstance(v, tuple | list):
            return [plain(x) for x in v]
        if isinstance(v, Mapping):
            return {str(k): plain(x) for k, x in v.items()}
        if isinstance(v, Path):
            return str(v)
        return v

    return {f.name: plain(getattr(config, f.name)) for f in dataclasses.fields(config)}
