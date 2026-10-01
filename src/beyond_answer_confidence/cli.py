"""The ``beyond-answer-confidence`` command line.

Subcommands::

    list                      experiments and analyses with summaries
    run <experiment>          collect (cache first), score, analyse, write
    analyse <analysis>        run an offline analysis (no requests)
    verify [experiment ...]   rebuild every request and check cache coverage
    estimate <experiment>     requests missing from the cache, token estimate
    report <name>             write tables and figures from the outputs
    export [name ...]         package per-item results without dataset text
    check-mirrors             compare dataset mirrors with independent copies

Runs are offline by default: requests are answered from the cache only.
``run --live`` sends the missing requests to the paid API and requires a
positive ``--max-input-tokens`` budget. The API key is read by the SDK from
the environment; this module never reads it.

Progress goes to the log (stderr); the final result of each subcommand is
written to stdout as JSON.
"""

import argparse
import functools
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from beyond_answer_confidence import __version__
from beyond_answer_confidence.json_output import dumps

logger = logging.getLogger("beyond_answer_confidence.cli")

QUIET_LOGGERS = ("typesafe_sdk", "httpx", "httpcore", "httpx2", "httpcore2")
"""Loggers pinned at WARNING: their debug output can include request headers."""


class CliError(Exception):
    """A user error: reported without a traceback, exit status 2."""


def configure_logging(verbosity: int) -> None:
    """Configure logging for a command-line run.

    Args:
        verbosity: ``-v`` count minus ``-q`` count (0 = INFO).
    """
    level = {-2: logging.ERROR, -1: logging.WARNING, 0: logging.INFO}.get(
        verbosity, logging.DEBUG if verbosity > 0 else logging.ERROR
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def emit(obj: Any) -> None:
    """Write a result to stdout as JSON.

    Args:
        obj: JSON-serialisable result.
    """
    sys.stdout.write(dumps(obj, indent=2, default=str) + "\n")


def concurrency_arg(value: str) -> int:
    """Parse ``--concurrency`` (an integer from 1 to 64).

    Args:
        value: The option value.

    Returns:
        The concurrency.

    Raises:
        argparse.ArgumentTypeError: If it is not an integer in range.
    """
    from beyond_answer_confidence.backends.cache import MAX_CONCURRENCY

    try:
        n = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from err
    if not 1 <= n <= MAX_CONCURRENCY:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_CONCURRENCY}")
    return n


API_KEY_ENV = "TYPESAFE_API_KEY"  # pragma: allowlist secret (a variable name)
"""Environment variable the SDK reads the key from (only its presence is
checked here; the value is never read into this package)."""


def api_key_present() -> bool:
    """Return whether the API key variable is set and non-empty.

    Only presence and a non-zero length are checked; the value is neither
    stored nor logged.

    Returns:
        Whether a key is configured.
    """
    return len(os.environ.get(API_KEY_ENV, "")) > 0


# --- settings and configuration ---------------------------------------------


def _settings(args: argparse.Namespace, **extra: Any) -> Any:
    from beyond_answer_confidence.settings import Settings

    return Settings.from_env(
        data_dir=getattr(args, "data_dir", None),
        output_dir=getattr(args, "output_dir", None),
        model=getattr(args, "model", None),
        **extra,
    )


def config_path(
    name: str, given: Path | None, config_dir: Path = Path("configs")
) -> Path | None:
    """Choose the configuration file of a run.

    Args:
        name: Experiment or analysis name.
        given: ``--config`` value, if any.
        config_dir: Directory searched for ``<name>.toml`` when no file is
            given (relative to the working directory).

    Returns:
        The file, or ``None`` for the built-in defaults.

    Raises:
        CliError: If a given file does not exist.
    """
    if given is not None:
        if not given.is_file():
            raise CliError(f"config file not found: {given}")
        return given
    default = config_dir / f"{name}.toml"
    return default if default.is_file() else None


def _load(name: str) -> Any:
    from beyond_answer_confidence.experiments import registry

    try:
        return registry.load(name)
    except KeyError as err:
        raise CliError(str(err.args[0])) from err


def _config(obj: Any, path: Path | None, overrides: Sequence[str]) -> Any:
    from beyond_answer_confidence.config import load_config

    try:
        config = load_config(obj.config_type, path, overrides)
    except (ValueError, TypeError) as err:
        raise CliError(f"{obj.name}: {err}") from err
    logger.info("%s: config %s", obj.name, path or "(defaults)")
    return config


def _experiment(name: str) -> Any:
    from beyond_answer_confidence.experiments.base import Experiment

    obj = _load(name)
    if not isinstance(obj, Experiment):
        raise CliError(f"{name} is an analysis; use `analyse {name}`")
    return obj


# --- subcommands ------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    """List experiments and analyses.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.
    """
    from beyond_answer_confidence.experiments import registry
    from beyond_answer_confidence.experiments.base import Experiment

    out = [
        {
            "name": o.name,
            "kind": "experiment" if isinstance(o, Experiment) else "analysis",
            "summary": o.summary,
        }
        for o in registry.load_all()
    ]
    emit(out)
    return 0


def budget_stops(summary: Any) -> int:
    """Count requests refused by the token budget anywhere in a run summary.

    Summaries report errors as ``errors: {type: count}`` (per part for
    experiments with several parts) or as an ``error_types`` list.

    Args:
        summary: A run summary.

    Returns:
        The number of refused requests (at least 1 if only a type list
        mentions the budget error).
    """
    name = "BudgetExceededError"
    if isinstance(summary, list):
        return sum(budget_stops(v) for v in summary)
    if not isinstance(summary, dict):
        return 0
    total = 0
    for k, v in summary.items():
        if k == name and isinstance(v, int) and not isinstance(v, bool):
            total += v
        elif k == "error_types" and isinstance(v, list) and name in v:
            total += 1
        elif isinstance(v, dict | list):
            total += budget_stops(v)
    return total


def _bookkeeping(name: str, settings: Any, summary: Any) -> dict[str, Any]:
    keys = ("new_api_calls", "new_input_tokens", "errors", "fatal")
    info = {"name": name, "output_dir": str(settings.out(name))}
    if isinstance(summary, dict):
        info |= {k: summary[k] for k in keys if k in summary}
    return info


def cmd_run(args: argparse.Namespace) -> int:
    """Run one experiment.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status (1 after a fatal API error or when the budget stopped
        calls).

    Raises:
        CliError: On a missing budget or API key for a live run, or a bad
            configuration.
    """
    if args.live and (args.max_input_tokens is None or args.max_input_tokens <= 0):
        raise CliError("--live requires --max-input-tokens N with N > 0")
    if args.live and not api_key_present():
        raise CliError(f"--live requires the {API_KEY_ENV} environment variable")
    exp = _experiment(args.name)
    config = _config(exp, config_path(exp.name, args.config), args.set)
    if args.live:
        logger.warning(
            "LIVE RUN: requests missing from the cache will be sent to the paid "
            "API (budget: %s new input tokens; stops after HTTP 401/402/403)",
            f"{args.max_input_tokens:,}",
        )
    elif args.max_input_tokens:
        logger.warning("--max-input-tokens is ignored without --live")
    settings = _settings(
        args,
        live=bool(args.live),
        max_input_tokens=args.max_input_tokens if args.live else 0,
        concurrency=args.concurrency,
    )
    summary = exp.run(settings, config)
    info = _bookkeeping(exp.name, settings, summary)
    emit(info)
    refused = budget_stops(summary)
    if refused:
        logger.warning(
            "requests not sent because the input-token budget was reached "
            "(%d reported); rerun with a larger --max-input-tokens to continue",
            refused,
        )
    return 1 if info.get("fatal") or refused else 0


def cmd_analyse(args: argparse.Namespace) -> int:
    """Run one offline analysis.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.

    Raises:
        CliError: If the name is an experiment.
    """
    from beyond_answer_confidence.experiments.base import Analysis

    obj = _load(args.name)
    if not isinstance(obj, Analysis):
        raise CliError(f"{args.name} is an experiment; use `run {args.name}`")
    config = _config(obj, config_path(obj.name, args.config), args.set)
    settings = _settings(args)
    summary = obj.run(settings, config)
    emit(_bookkeeping(obj.name, settings, summary))
    return 0


def _targets(
    names: Sequence[str], settings: Any, overrides: Sequence[str]
) -> list[Any]:
    from beyond_answer_confidence.experiments import registry
    from beyond_answer_confidence.reproduce import Target

    targets = []
    for name in names:
        exp = _experiment(name)
        config = _config(exp, config_path(name, None), overrides)
        targets.append(
            Target(
                name,
                settings.cache_file(registry.cache_name(name)),
                functools.partial(exp.requests, settings, config),
            )
        )
    return targets


def cmd_verify(args: argparse.Namespace) -> int:
    """Rebuild every request and check that the caches answer them.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status (1 if any request is missing).
    """
    from beyond_answer_confidence.experiments import registry
    from beyond_answer_confidence.reproduce import verify

    settings = _settings(args)
    names = args.names or registry.experiment_names()
    result = verify(_targets(names, settings, ()), settings.model)
    emit(result)
    return 0 if result["ok"] else 1


def cmd_estimate(args: argparse.Namespace) -> int:
    """Count requests missing from the cache and estimate their input tokens.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.
    """
    from beyond_answer_confidence.experiments import registry
    from beyond_answer_confidence.reproduce import Target, estimate_all

    settings = _settings(args)
    exp = _experiment(args.name)
    config = _config(exp, config_path(exp.name, args.config), args.set)
    target = Target(
        exp.name,
        settings.cache_file(registry.cache_name(exp.name)),
        lambda: exp.requests(settings, config),
    )
    emit(estimate_all([target], settings.model))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Write tables and figures of one experiment or analysis.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.

    Raises:
        CliError: If no reporter exists or the outputs are missing.
    """
    from beyond_answer_confidence.reporting import report, reportable

    if args.name not in reportable():
        raise CliError(f"no reporter for {args.name!r}; available: {reportable()}")
    try:
        written = report(args.name, _settings(args), figures=not args.no_figures)
    except FileNotFoundError as err:
        raise CliError(f"outputs missing, run the experiment first: {err}") from err
    emit({"name": args.name, "files": [str(p) for p in written]})
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Package per-item results without dataset text.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.

    Raises:
        CliError: If a file could carry dataset text or the target is unsafe.
    """
    from beyond_answer_confidence import export
    from beyond_answer_confidence.experiments import registry

    settings = _settings(args)
    names = args.names or registry.names()
    if unknown := sorted(set(names) - set(registry.names())):
        raise CliError(f"unknown experiment or analysis: {unknown}")
    out = args.out or settings.output_dir / "export"
    tar_path = out.with_name(out.name + ".tar.gz")
    if args.tar and tar_path.exists() and not args.force:
        raise CliError(f"export refused: {tar_path} exists; pass --force to replace it")
    distractors = settings.cache_dir / "distractors"
    try:
        manifest = export.build(
            settings.output_dir,
            names,
            out,
            distractors=distractors
            if distractors.is_dir() and not args.no_distractors
            else None,
        )
    except (export.UnsafeExportError, FileExistsError, ValueError) as err:
        raise CliError(f"export refused: {err}") from err
    info: dict[str, Any] = {
        "out": str(out),
        "files": len(manifest["files"]),
        "bytes": sum(f["bytes"] for f in manifest["files"]),
    }
    if args.tar:
        info["tar"] = str(export.make_tar(out, force=args.force))
    emit(info)
    return 0


def cmd_check_mirrors(args: argparse.Namespace) -> int:
    """Compare dataset mirrors with independent copies (needs network).

    Args:
        args: Parsed arguments.

    Returns:
        Exit status (1 if the mirrors are inconsistent).
    """
    from beyond_answer_confidence.data.mirrors import check_mirrors

    result = check_mirrors(_settings(args).data_dir)
    emit(result)
    return 0 if result.get("consistent") else 1


# --- parser -----------------------------------------------------------------


def _paths(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data-dir",
        type=Path,
        help="datasets and caches (default: $BEYOND_ANSWER_CONFIDENCE_DATA_DIR or ./data)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        help="outputs (default: $BEYOND_ANSWER_CONFIDENCE_OUTPUT_DIR or ./outputs)",
    )


def _model(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", help="model alias (part of every cache key)")


def _config_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        type=Path,
        help="TOML configuration (default: configs/<name>.toml if present)",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one configuration value (repeatable)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="beyond-answer-confidence",
        description="Probe whether a decision API's probabilities track what it knows.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-q", "--quiet", action="count", default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list experiments and analyses")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("run", help="run an experiment (offline unless --live)")
    p.add_argument("name")
    _config_opts(p)
    _paths(p)
    _model(p)
    p.add_argument(
        "--live",
        action="store_true",
        help="send uncached requests to the paid API (needs --max-input-tokens)",
    )
    p.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="budget of new input tokens for a live run",
    )
    p.add_argument(
        "--concurrency",
        type=concurrency_arg,
        default=8,
        help="simultaneous API calls (1-64)",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyse", help="run an offline analysis")
    p.add_argument("name")
    _config_opts(p)
    _paths(p)
    _model(p)
    p.set_defaults(func=cmd_analyse)

    p = sub.add_parser("verify", help="rebuild requests and check cache coverage")
    p.add_argument("names", nargs="*", help="experiments (default: all)")
    _paths(p)
    _model(p)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("estimate", help="uncached requests and input-token estimate")
    p.add_argument("name")
    _config_opts(p)
    _paths(p)
    _model(p)
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("report", help="write tables and figures from the outputs")
    p.add_argument("name")
    p.add_argument("--no-figures", action="store_true", help="tables only")
    _paths(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("export", help="package per-item results without dataset text")
    p.add_argument("names", nargs="*", help="experiments and analyses (default: all)")
    p.add_argument(
        "--out", type=Path, help="export directory (default: <output_dir>/export)"
    )
    p.add_argument("--tar", action="store_true", help="also write <out>.tar.gz")
    p.add_argument(
        "--force", action="store_true", help="replace an existing <out>.tar.gz"
    )
    p.add_argument(
        "--no-distractors",
        action="store_true",
        help="leave out the saved distractor choices",
    )
    _paths(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("check-mirrors", help="compare dataset mirrors (network)")
    _paths(p)
    p.set_defaults(func=cmd_check_mirrors)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        Exit status.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose - args.quiet)
    try:
        status: int = args.func(args)
    except CliError as err:
        logger.error("%s", err)
        return 2
    return status


if __name__ == "__main__":
    raise SystemExit(main())
