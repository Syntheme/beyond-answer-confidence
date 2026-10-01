import json
import logging
from pathlib import Path
from typing import Any

import pytest

from beyond_answer_confidence import cli
from beyond_answer_confidence.data import mirrors
from beyond_answer_confidence.experiments import registry
from beyond_answer_confidence.experiments.registry import Entry

TOY = (
    Entry("toy_experiment", "toy_experiment", "EXPERIMENT"),
    Entry("toy_analysis", "toy_experiment", "ANALYSIS"),
)


@pytest.fixture
def toy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Register only the toy objects and work in an empty directory."""
    monkeypatch.setattr(registry, "ENTRIES", TOY)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BEYOND_ANSWER_CONFIDENCE_DATA_DIR", raising=False)
    monkeypatch.delenv("BEYOND_ANSWER_CONFIDENCE_OUTPUT_DIR", raising=False)
    return tmp_path


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, Any]:
    code = cli.main(["-q", *argv])
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else None


def paths(tmp: Path) -> list[str]:
    return ["--data-dir", str(tmp / "d"), "--output-dir", str(tmp / "o")]


def test_list_real_registry(capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(capsys, "list")
    assert code == 0
    names = [o["name"] for o in out]
    assert names == registry.names()
    kinds = {o["name"]: o["kind"] for o in out}
    assert kinds["knowledge_dial"] == "experiment"
    assert kinds["error_detection"] == "analysis"


def test_offline_run_answers_nothing(
    toy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(capsys, "run", "toy_experiment", *paths(toy))
    assert code == 0
    assert out["new_api_calls"] == 0
    assert out["output_dir"] == str(toy / "o" / "toy_experiment")
    summary = json.loads((toy / "o/toy_experiment/summary.json").read_text())
    assert summary["complete_items"] == 0


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A placeholder key: live runs only check that the variable is set."""
    monkeypatch.setenv(cli.API_KEY_ENV, "placeholder")  # pragma: allowlist secret


def test_live_run_needs_api_key(
    toy: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(cli.API_KEY_ENV, raising=False)
    argv = ["run", "toy_experiment", *paths(toy), "--live", "--max-input-tokens", "9"]
    assert cli.main(argv) == 2
    monkeypatch.setenv(cli.API_KEY_ENV, "")
    assert cli.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert cli.API_KEY_ENV in captured.err
    assert not (toy / "d").exists()


@pytest.mark.parametrize("value", ["0", "-1", "65", "x"])
def test_concurrency_is_validated(
    toy: Path, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "toy_experiment", "--concurrency", value])
    assert exc.value.code == 2
    assert "--concurrency" in capsys.readouterr().err


def test_live_run_stopped_by_budget_exits_1(
    toy: Path, api_key: None, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "run",
        "toy_experiment",
        *paths(toy),
        "--live",
        "--max-input-tokens",
        "150",
        "--concurrency",
        "1",
    ]
    code = cli.main(argv)
    captured = capsys.readouterr()
    assert code == 1
    assert 0 < json.loads(captured.out)["new_api_calls"] < 8
    assert "budget was reached" in captured.err


def test_budget_stops_are_found_in_any_summary_shape() -> None:
    assert cli.budget_stops({"errors": {"BudgetExceededError": 3}}) == 3
    assert cli.budget_stops({"a": {"errors": {"BudgetExceededError": 2}}}) == 2
    assert cli.budget_stops({"error_types": ["BudgetExceededError"]}) == 1
    assert cli.budget_stops({"errors": {"OfflineError": 5}, "x": [1, "y"]}) == 0
    assert cli.budget_stops([{"errors": {"BudgetExceededError": 1}}, 7]) == 1


def test_live_run_needs_budget(toy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "toy_experiment", "--live"]) == 2
    assert cli.main(["run", "toy_experiment", "--live", "--max-input-tokens", "0"]) == 2
    assert capsys.readouterr().out == ""


def test_live_run_warns_and_fills_cache(
    toy: Path, api_key: None, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "run",
        "toy_experiment",
        *paths(toy),
        "--live",
        "--max-input-tokens",
        "10000",
    ]
    code = cli.main(argv)
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert code == 0
    assert out["new_api_calls"] == 8
    assert "LIVE RUN" in captured.err
    assert (toy / "d/cache/toy_cache.jsonl").exists()
    # Offline afterwards: all answered from the cache.
    code, out = run(capsys, "run", "toy_experiment", *paths(toy))
    assert out["new_api_calls"] == 0


def test_budget_without_live_is_ignored(
    toy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["run", "toy_experiment", *paths(toy), "--max-input-tokens", "5"])
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["new_api_calls"] == 0
    assert "ignored without --live" in captured.err


def test_config_file_and_overrides(
    toy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (toy / "configs").mkdir()
    (toy / "configs/toy_experiment.toml").write_text("items = 3\n")
    run(capsys, "run", "toy_experiment", *paths(toy))
    summary = json.loads((toy / "o/toy_experiment/summary.json").read_text())
    assert summary["config"]["items"] == 3
    run(capsys, "run", "toy_experiment", *paths(toy), "--set", "items=2")
    summary = json.loads((toy / "o/toy_experiment/summary.json").read_text())
    assert summary["config"]["items"] == 2
    other = toy / "other.toml"
    other.write_text("items = 1\n")
    run(capsys, "run", "toy_experiment", *paths(toy), "--config", str(other))
    summary = json.loads((toy / "o/toy_experiment/summary.json").read_text())
    assert summary["config"]["items"] == 1


def test_config_errors(toy: Path) -> None:
    assert cli.main(["run", "toy_experiment", "--config", "missing.toml"]) == 2
    assert cli.main(["run", "toy_experiment", "--set", "nope=1"]) == 2
    assert cli.main(["run", "toy_experiment", "--set", "novalue"]) == 2


def test_wrong_kind_and_unknown_names(toy: Path) -> None:
    assert cli.main(["run", "toy_analysis"]) == 2
    assert cli.main(["analyse", "toy_experiment"]) == 2
    assert cli.main(["run", "nothing"]) == 2
    assert cli.main(["verify", "toy_analysis"]) == 2


def test_analyse(toy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(capsys, "analyse", "toy_analysis", *paths(toy), "--set", "items=9")
    assert code == 0
    assert out["name"] == "toy_analysis"
    assert json.loads((toy / "o/toy_analysis/summary.json").read_text()) == {"items": 9}


def test_verify_and_estimate(
    toy: Path, api_key: None, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(capsys, "verify", *paths(toy))
    assert code == 1
    assert out["experiments"]["toy_experiment"]["missing"] == 8
    code, out = run(capsys, "estimate", "toy_experiment", *paths(toy))
    assert code == 0
    assert out["new_requests"] == 8
    assert out["estimated_input_tokens"] > 0
    cli.main(
        ["run", "toy_experiment", *paths(toy), "--live", "--max-input-tokens", "10000"]
    )
    capsys.readouterr()
    code, out = run(capsys, "verify", "toy_experiment", *paths(toy))
    assert (code, out["ok"], out["requests"]) == (0, True, 8)
    code, out = run(capsys, "estimate", "toy_experiment", *paths(toy))
    assert out["new_requests"] == 0


def test_verify_model_is_part_of_the_key(
    toy: Path, api_key: None, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(
        ["run", "toy_experiment", *paths(toy), "--live", "--max-input-tokens", "10000"]
    )
    capsys.readouterr()
    code, out = run(capsys, "verify", *paths(toy), "--model", "another")
    assert code == 1
    assert out["missing"] == 8


def test_env_directories(
    toy: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BEYOND_ANSWER_CONFIDENCE_OUTPUT_DIR", str(toy / "envout"))
    _, out = run(capsys, "run", "toy_experiment")
    assert out["output_dir"] == str(toy / "envout" / "toy_experiment")


def test_export(toy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(capsys, "run", "toy_experiment", *paths(toy))
    (toy / "d/cache/distractors").mkdir(parents=True)
    (toy / "d/cache/distractors/x.json").write_text("[1, 2]")
    code, out = run(capsys, "export", *paths(toy), "--tar")
    assert code == 0
    assert Path(out["tar"]).exists()
    manifest = json.loads((toy / "o/export/MANIFEST.json").read_text())
    assert {f["path"] for f in manifest["files"]} >= {
        "toy_experiment/summary.json",
        "distractors/x.json",
    }
    code, out = run(capsys, "export", "toy_experiment", *paths(toy), "--no-distractors")
    manifest = json.loads((toy / "o/export/MANIFEST.json").read_text())
    assert all(not f["path"].startswith("distractors") for f in manifest["files"])
    # An existing archive is not replaced without --force.
    assert cli.main(["-q", "export", *paths(toy), "--tar"]) == 2
    code, out = run(capsys, "export", *paths(toy), "--tar", "--force")
    assert code == 0
    assert Path(out["tar"]).exists()


def test_export_refusals(toy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["export", "nothing", *paths(toy)]) == 2
    rows = toy / "o/toy_experiment/rows.jsonl"
    rows.parent.mkdir(parents=True)
    rows.write_text(json.dumps({"unit": "x", "text": "a message"}) + "\n")
    assert cli.main(["export", *paths(toy)]) == 2
    assert not (toy / "o/export").exists()


def test_report(
    toy: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli.main(["report", "toy_experiment", *paths(toy)]) == 2
    assert cli.main(["report", "benchmarks", *paths(toy)]) == 2  # no outputs yet
    out_dir = toy / "o/benchmarks"
    out_dir.mkdir(parents=True)
    (out_dir / "summary.json").write_text(json.dumps({"results": {"calibration": {}}}))
    code, out = run(capsys, "report", "benchmarks", *paths(toy), "--no-figures")
    assert code == 0
    assert out["files"] == [str(out_dir / "tables.md")]


def test_check_mirrors(
    toy: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []

    def fake(data_dir: Path) -> dict[str, Any]:
        seen.append(data_dir)
        return {"consistent": len(seen) == 1}

    monkeypatch.setattr(mirrors, "check_mirrors", fake)
    code, out = run(capsys, "check-mirrors", "--data-dir", str(toy / "d"))
    assert (code, out) == (0, {"consistent": True})
    assert seen == [toy / "d"]
    code, _ = run(capsys, "check-mirrors")
    assert code == 1


def test_run_exit_status_after_fatal_error(
    toy: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses

    import toy_experiment

    exp = toy_experiment.EXPERIMENT
    fatal = dataclasses.replace(
        exp, run=lambda s, c: {"fatal": "HTTP 402", "new_api_calls": 1}
    )
    monkeypatch.setattr(toy_experiment, "EXPERIMENT", fatal)
    code, out = run(capsys, "run", "toy_experiment", *paths(toy))
    assert code == 1
    assert out["fatal"] == "HTTP 402"


def test_logging_levels() -> None:
    for v, level in (
        (-3, logging.ERROR),
        (-1, logging.WARNING),
        (0, logging.INFO),
        (2, logging.DEBUG),
    ):
        cli.configure_logging(v)
        assert logging.getLogger().level == level
        assert logging.getLogger("typesafe_sdk").level == logging.WARNING


def test_config_path(tmp_path: Path) -> None:
    assert cli.config_path("x", None, tmp_path) is None
    (tmp_path / "x.toml").write_text("")
    assert cli.config_path("x", None, tmp_path) == tmp_path / "x.toml"
    with pytest.raises(cli.CliError):
        cli.config_path("x", tmp_path / "nope.toml")


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out.strip()
