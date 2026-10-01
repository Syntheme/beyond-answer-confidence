import json
import re
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

BLOG = Path(__file__).resolve().parents[2] / "assets" / "blog"
sys.path.insert(0, str(BLOG))
import webflow  # noqa: E402
import webflow_api  # noqa: E402

wf: ModuleType = webflow
api: ModuleType = webflow_api

TOKEN = "fake-token"  # noqa: S105
REF = "v9.9.9"
BASE = wf.ASSET_BASE.format(ref=REF)


class FakeWebflow:
    """Enough of the Webflow Data API v2 for the commands, in memory."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.fields = [
            {"slug": "name", "type": "PlainText", "displayName": "Name"},
            {"slug": "slug", "type": "PlainText", "displayName": "Slug"},
            {"slug": "post-body", "type": "RichText", "displayName": "Post body"},
            {"slug": "summary", "type": "PlainText", "displayName": "Summary"},
        ]
        self.items: dict[str, dict[str, Any]] = {}
        self.registered: list[dict[str, Any]] = []
        self.custom_code: dict[str, list[dict[str, Any]]] = {"sites/s1": []}
        self.mangle = False

    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, bytes]:
        assert headers["Authorization"] == f"Bearer {TOKEN}"
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.removeprefix("/v2/")
        data = json.loads(body) if body else None
        self.calls.append((method, path, data))
        out = self.route(method, path, data, urllib.parse.parse_qs(parsed.query))
        return (200, json.dumps(out).encode()) if out is not None else (404, b"{}")

    def route(  # noqa: C901
        self, method: str, path: str, data: Any, query: dict[str, list[str]]
    ) -> Any:
        if (method, path) == ("GET", "sites"):
            return {"sites": [{"id": "s1", "displayName": "Test", "shortName": "t"}]}
        if (method, path) == ("GET", "sites/s1/collections"):
            return {
                "collections": [{"id": "c1", "displayName": "Posts", "slug": "blog"}]
            }
        if (method, path) == ("GET", "collections/c1"):
            return {"id": "c1", "fields": self.fields}
        if (method, path) == ("GET", "sites/s1/pages"):
            pages = [
                {"id": "p1", "title": "Home"},
                {"id": "p2", "title": "Post", "collectionId": "c1"},
            ]
            return {"pages": pages, "pagination": {"total": 2}}
        if (method, path) == ("GET", "collections/c1/items"):
            offset = int(query["offset"][0])
            items = list(self.items.values())
            return {
                "items": items[offset : offset + 100],
                "pagination": {"total": len(items)},
            }
        if (method, path) == ("POST", "collections/c1/items"):
            item_id = f"i{len(self.items) + 1}"
            self.items[item_id] = {
                "id": item_id,
                "isDraft": data["isDraft"],
                "fieldData": dict(data["fieldData"]),
            }
            return self.items[item_id]
        if m := re.fullmatch(r"collections/c1/items/(\w+)", path):
            item = self.items[m.group(1)]
            if method == "PATCH":
                item["fieldData"].update(data["fieldData"])
                if self.mangle:
                    item["fieldData"]["post-body"] += "<br>"
            return item
        if (method, path) == ("GET", "sites/s1/registered_scripts"):
            return {"registeredScripts": self.registered}
        if (method, path) == ("POST", "sites/s1/registered_scripts/hosted"):
            self.registered.append({**data, "id": "bacblog"})
            return self.registered[-1]
        if (m := re.fullmatch(r"((?:sites|pages)/\w+)/custom_code", path)) and m.group(
            1
        ) in self.custom_code:
            if method == "PUT":
                self.custom_code[m.group(1)] = data["scripts"]
            return {"scripts": self.custom_code[m.group(1)]}
        return None

    def writes(self) -> list[tuple[str, str]]:
        return [(m, p) for m, p, _ in self.calls if m != "GET"]


@pytest.fixture(scope="module")
def served() -> dict[str, str]:
    files: dict[str, str] = wf.build_files(BASE)[0]
    return {BASE + name: text for name, text in files.items()}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeWebflow:
    monkeypatch.setenv(api.TOKEN_ENV, TOKEN)
    return FakeWebflow()


def run(
    fake: FakeWebflow,
    served: dict[str, str],
    capsys: pytest.CaptureFixture[str],
    *argv: str,
) -> tuple[int, str]:
    def fetch(url: str) -> bytes:
        if url not in served:
            raise api.ApiError(f"{url}: HTTP 404")
        return served[url].encode()

    code = api.main(list(argv), send=fake.send, fetch=fetch)
    out, err = capsys.readouterr()
    assert TOKEN not in out + err
    return code, out + err


def split_blocks(text: str) -> list[str]:
    return re.split(r"\n(?=<div data-rt-embed-type=)", text.strip())


PUSH = (
    "push",
    "--collection",
    "c1",
    "--asset-ref",
    REF,
    "--body-field",
    "post-body",
    "--summary-field",
    "summary",
)


# -- build --------------------------------------------------------------------


def test_committed_webflow_files_match_sources() -> None:
    assert wf.main(["--check"]) == 0


def test_scope_css_scopes_rules_and_drops_page_and_theme_rules() -> None:
    css = (
        ":root { --a: 1; } body { x: 1; } html { y: 1; } h2, p { z: 2; } "
        '[data-theme="dark"] p { w: 1; } /* note */ @media (max-width: 5px) { main { v: 1; } a { u: 1; } }'
    )
    assert wf.scope_css(css).splitlines() == [
        ".bac-post { --a: 1; }",
        ".bac-post { x: 1; }",
        ".bac-post h2, .bac-post p { z: 2; }",
        "@media (max-width: 5px) {",
        ".bac-post a { u: 1; }",
        "}",
    ]


def test_diagrams_become_images_with_resolved_colours() -> None:
    svg = '<svg viewBox="0 0 300 100" aria-label="A diagram"><rect fill="var(--ink)"/></svg>'
    body, files = wf.extract_diagrams(
        f"<p>x</p>{svg}", {"--ink": "#111"}, "https://cdn/"
    )
    ((name, text),) = files.items()
    assert (
        body
        == f'<p>x</p><img src="https://cdn/{name}" alt="A diagram" width="300" height="100" loading="lazy">'
    )
    assert 'fill="#111"' in text
    assert 'xmlns="http://www.w3.org/2000/svg"' in text
    with pytest.raises(wf.build.BuildError, match="unknown colour"):
        wf.extract_diagrams(svg, {}, "")


def test_page_links_point_at_post_urls() -> None:
    body = '<a href="part-2.html">next</a> <a href="part-2.html#s3">s3</a> <a href="#n1">1</a>'
    assert wf.page_links(body, {"part-2.html": "/blog/two"}) == (
        '<a href="/blog/two">next</a> <a href="/blog/two#s3">s3</a> <a href="#n1">1</a>'
    )
    with pytest.raises(wf.build.BuildError, match="no URL"):
        wf.page_links(body, {})


def test_embed_blocks_split_between_top_level_elements() -> None:
    paras = [f"<p>{'word ' * 30}{i}</p>" for i in range(40)]
    body = '<div class="bac-post">\n' + "\n".join(paras) + "\n</div>\n"
    blocks = split_blocks(wf.embed_blocks(body, limit=1_000))
    assert len(blocks) > 5
    assert all(len(b) <= 1_000 for b in blocks)
    inner = [
        re.fullmatch(
            r"<div data-rt-embed-type='true'><div class=\"bac-post\">(.*)</div></div>",
            b,
            re.S,
        )
        for b in blocks
    ]
    assert all(inner)
    assert re.findall(r"<p>.*?</p>", "".join(m.group(1) for m in inner if m)) == paras
    with pytest.raises(wf.build.BuildError, match="larger than"):
        wf.embed_blocks(body, limit=200)


def test_built_pages_fit_rich_text_blocks_and_link_to_posts() -> None:
    files, rich = wf.build_files(
        BASE, {"index.html": "/a", "part-1.html": "/b", "part-2.html": "/c"}
    )
    for text in rich.values():
        assert all(len(b) <= wf.EMBED_LIMIT for b in split_blocks(text))
        assert "<svg" not in text
        assert not re.search(r'href="[a-z0-9-]+\.html', text)
    assert 'href="/c"' in rich["part-1.html"]
    assert all(
        f'src="{BASE}{n}"' in rich["index.html"]
        for n in files
        if n.startswith("diagrams/")
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_blog_js_does_nothing_without_the_post() -> None:
    stub = (
        "globalThis.window = globalThis;"
        "globalThis.document = {currentScript: null, readyState: 'complete', querySelector: () => null};"
    )
    check = "if (window.BAC_BLOG_DATA !== undefined) throw new Error('ran');"
    js = (BLOG / "webflow" / "blog.js").read_text(encoding="utf-8")
    subprocess.run(["node", "-e", stub + js + check], check=True, timeout=30)  # noqa: S603, S607


# -- client -------------------------------------------------------------------


def test_client_never_publishes_and_writes_only_when_allowed() -> None:
    sent: list[str] = []

    def send(
        method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, bytes]:
        sent.append(url)
        return 200, b"{}"

    client = api.Client(TOKEN, send, writes=True)
    for path in (
        "sites/s1/publish",
        "collections/c1/items/live",
        "collections/c1/items/publish",
    ):
        with pytest.raises(api.ApiError, match="publish"):
            client.request("POST", path, {})
    with pytest.raises(api.ApiError, match="dry run"):
        api.Client(TOKEN, send).request("PATCH", "collections/c1/items/i1", {})
    assert sent == []
    assert TOKEN not in repr(client)


def test_client_waits_on_rate_limit_and_reports_errors() -> None:
    statuses = [429, 429, 200, 500]
    waits: list[float] = []

    def send(
        method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, bytes]:
        return statuses.pop(0), b'{"ok": true}'

    client = api.Client(TOKEN, send, sleep=waits.append)
    assert client.get("sites") == {"ok": True}
    assert waits == [60, 60]
    with pytest.raises(api.ApiError, match="HTTP 500"):
        client.get("sites")


# -- commands -----------------------------------------------------------------


def test_discover_lists_fields_and_template_pages(
    fake: FakeWebflow, served: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(fake, served, capsys, "discover")
    assert code == 0
    assert "collection c1  Posts  /blog/" in out
    assert re.search(r"field post-body\s+RichText", out)
    assert "template page p2  Post  (collection c1)" in out
    assert "p1" not in out
    assert fake.writes() == []


def test_push_dry_run_sends_nothing(
    fake: FakeWebflow, served: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(fake, served, capsys, *PUSH)
    assert code == 0
    assert out.count("would create draft") == 2
    assert "dry run: nothing sent" in out
    assert fake.writes() == []


def test_push_refuses_moving_refs_and_unchecked_applies(
    fake: FakeWebflow, served: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(
        fake, served, capsys, "push", "--collection", "c1", "--asset-ref", "main"
    )
    assert (code, "use a tag" in out) == (1, True)
    code, out = run(fake, served, capsys, *PUSH, "--no-asset-check", "--apply")
    assert (code, "dry runs only" in out) == (1, True)
    assert fake.calls == []


def test_push_needs_the_assets_at_the_ref(
    fake: FakeWebflow, served: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    stale = {
        url: text + " " if url.endswith("blog.js") else text
        for url, text in served.items()
    }
    code, out = run(fake, stale, capsys, *PUSH)
    assert code == 1
    assert "blog.js: served file differs" in out
    code, out = run(fake, stale, capsys, *PUSH, "--no-asset-check")
    assert code == 0


def test_push_checks_the_collection_fields(
    fake: FakeWebflow, served: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(fake, served, capsys, *PUSH[:-4])
    assert (code, "no field 'body'" in out) == (1, True)
    fake.fields[2]["type"] = "PlainText"
    code, out = run(fake, served, capsys, *PUSH)
    assert (code, "not RichText" in out) == (1, True)


def test_push_creates_drafts_then_updates_only_changed_fields(
    fake: FakeWebflow, served: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(fake, served, capsys, *PUSH, "--apply")
    assert code == 0
    assert out.count("created draft") == 2
    assert all(item["isDraft"] for item in fake.items.values())
    first = fake.items["i1"]["fieldData"]
    assert first["slug"] == api.SLUGS["part-1.html"]
    assert f'href="/blog/{api.SLUGS["part-2.html"]}"' in first["post-body"]
    assert first["summary"].startswith("Part 1 of 2.")

    code, out = run(fake, served, capsys, *PUSH)
    assert out.count("unchanged") == 2

    first["post-body"] = "<p>edited in Webflow</p>"
    fake.calls.clear()
    code, out = run(fake, served, capsys, *PUSH, "--diff")
    assert "would update" in out
    assert "-<p>edited in Webflow</p>" in out
    code, out = run(fake, served, capsys, *PUSH, "--apply")
    assert code == 0
    patches = [d for m, _, d in fake.calls if m == "PATCH"]
    assert patches == [
        {"fieldData": {"post-body": fake.items["i1"]["fieldData"]["post-body"]}}
    ]
    assert not any("publish" in p for _, p in fake.writes())


def test_push_reports_items_stored_differently(
    fake: FakeWebflow, served: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    run(fake, served, capsys, *PUSH, "--apply", "--page", "part-1.html")
    fake.items["i1"]["fieldData"]["post-body"] = "old"
    fake.mangle = True
    code, out = run(fake, served, capsys, *PUSH, "--apply", "--page", "part-1.html")
    assert code == 1
    assert "stored differently" in out


def test_script_registers_once_and_keeps_other_scripts(
    fake: FakeWebflow, served: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    other = {"id": "analytics", "location": "header", "version": "1.0.0"}
    fake.custom_code["sites/s1"] = [other]
    script = ("script", "--site", "s1", "--asset-ref", REF)

    code, out = run(fake, served, capsys, *script)
    assert code == 0
    assert "would register" in out
    assert fake.writes() == []

    code, out = run(fake, served, capsys, *script, "--apply")
    assert code == 0
    (reg,) = fake.registered
    assert reg["hostedLocation"] == BASE + "blog.js"
    assert reg["integrityHash"] == api.sri(served[BASE + "blog.js"])
    assert reg["version"] == "9.9.9"
    entry = {"id": "bacblog", "location": "footer", "version": "9.9.9"}
    assert fake.custom_code["sites/s1"] == [other, entry]

    code, out = run(fake, served, capsys, *script, "--apply")
    assert code == 0
    assert len(fake.registered) == 1
    assert fake.custom_code["sites/s1"] == [other, entry]

    fake.registered[0]["integrityHash"] = "sha384-other"
    code, out = run(fake, served, capsys, *script)
    assert (code, "use a new version" in out) == (1, True)


def test_missing_token_is_an_error(
    fake: FakeWebflow,
    served: dict[str, str],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(api.TOKEN_ENV)
    code, out = run(fake, served, capsys, "discover")
    assert (code, "no token" in out) == (1, True)


def test_push_keeps_titles_edited_in_webflow(
    fake: FakeWebflow, served: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    run(fake, served, capsys, *PUSH, "--apply")
    fake.items["i1"]["fieldData"]["name"] = "A title from marketing"
    code, out = run(fake, served, capsys, *PUSH, "--apply")
    assert code == 0
    assert fake.items["i1"]["fieldData"]["name"] == "A title from marketing"
    code, out = run(fake, served, capsys, *PUSH, "--update-meta")
    assert "would update" in out
    assert "name" in out
