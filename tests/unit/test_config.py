from dataclasses import dataclass
from pathlib import Path

import pytest

from beyond_answer_confidence.config import as_dict, load_config
from beyond_answer_confidence.settings import Settings


@dataclass(frozen=True)
class Demo:
    seed: int = 0
    rate: float = 0.5
    name: str = "x"
    flag: bool = False
    sizes: tuple[int, ...] = (1, 2)
    limit: int | None = None


def test_defaults() -> None:
    assert load_config(Demo) == Demo()


def test_toml_then_overrides(tmp_path: Path) -> None:
    path = tmp_path / "demo.toml"
    path.write_text('seed = 3\nsizes = [4, 5]\nname = "file"\n')
    cfg = load_config(Demo, path, ["name=cli", "flag=true", "sizes=7,8", "limit=9"])
    assert cfg == Demo(seed=3, rate=0.5, name="cli", flag=True, sizes=(7, 8), limit=9)
    assert load_config(Demo, None, ["limit=none"]).limit is None


@pytest.mark.parametrize("bad", [["nope=1"], ["seed"], ["flag=maybe"]])
def test_bad_overrides(bad: list[str]) -> None:
    with pytest.raises(ValueError, match=r"unknown|key=value|boolean"):
        load_config(Demo, None, bad)


def test_as_dict_is_json_friendly() -> None:
    assert as_dict(Demo()) == {
        "seed": 0,
        "rate": 0.5,
        "name": "x",
        "flag": False,
        "sizes": [1, 2],
        "limit": None,
    }


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BEYOND_ANSWER_CONFIDENCE_DATA_DIR", str(tmp_path / "d"))
    s = Settings.from_env(live=None, concurrency=2)
    assert s.data_dir == tmp_path / "d"
    assert s.cache_file("demo") == tmp_path / "d" / "cache" / "demo.jsonl"
    assert s.out("demo") == Path("outputs") / "demo"
    assert s.concurrency == 2
    assert s.live is False
    assert s.hf_dir == tmp_path / "d" / "hf"


SHIPPED = sorted((Path(__file__).resolve().parents[2] / "configs").glob("*.toml"))


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_shipped_configs_equal_the_defaults(path: Path) -> None:
    from beyond_answer_confidence.experiments import registry

    obj = registry.load(path.stem)
    assert load_config(obj.config_type, path, []) == obj.config_type()
