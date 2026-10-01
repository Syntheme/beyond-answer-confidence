import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

BUILD = Path(__file__).resolve().parents[2] / "assets" / "blog" / "build.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("blog_build", BUILD)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["blog_build"] = module
    spec.loader.exec_module(module)
    return module


blog = _load()


def test_committed_pages_match_sources() -> None:
    assert blog.main(["--check"]) == 0


def test_parts_split_the_charts_of_the_full_post() -> None:
    pages = blog.build()
    charts = {
        name: re.findall(r'<div class="chart" id="(\w+)"', text)
        for name, text in pages.items()
    }
    assert len(charts["index.html"]) == 18
    assert sorted(charts["part-1.html"] + charts["part-2.html"]) == sorted(
        charts["index.html"]
    )


def test_footnotes_are_numbered_in_reading_order() -> None:
    for name, text in blog.build().items():
        refs = [int(n) for n in re.findall(r'<a href="#n(\d+)" id="r\d+">', text)]
        notes = [int(n) for n in re.findall(r'<li id="n(\d+)">', text)]
        assert refs == list(range(1, len(refs) + 1)), name
        assert notes == refs, name


def test_sections_are_numbered_from_one_on_each_page() -> None:
    for name, text in blog.build().items():
        eyebrows = re.findall(r'<span class="eyebrow">(\d\d) · ', text)
        assert eyebrows == [f"{i:02d}" for i in range(1, len(eyebrows) + 1)], name


def test_undefined_footnote_is_an_error() -> None:
    with pytest.raises(blog.BuildError, match="not defined"):
        blog.number_notes('x<sup data-note="nope"></sup>', {})


def test_repeated_footnote_keeps_its_first_number() -> None:
    body, items, used = blog.number_notes(
        '<sup data-note="a"></sup><sup data-note="b"></sup><sup data-note="a"></sup>',
        {"a": "A", "b": "B"},
    )
    assert re.findall(r'href="#n(\d)"', body) == ["1", "2", "1"]
    assert used == {"a", "b"}
    assert items.count("<li") == 2
