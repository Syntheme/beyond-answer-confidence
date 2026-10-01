"""Put the Webflow build of the blog into a Webflow site through its Data API.

Three commands, all dry runs unless given ``--apply``:

- ``discover``: list the sites the token can see, with their collections and
  fields and their CMS template pages (read-only, has no ``--apply``);
- ``push``: create or update one CMS item per page, matched by slug. New
  items are created as drafts with title, slug, body and summary; existing
  items keep their draft/published state and get only a changed body (with
  ``--update-meta`` also the title and summary), so edits made in Webflow to
  the other fields stay. The dry run prints what would change;
  ``--apply`` sends it and reads each item back to check it was stored as sent;
- ``script``: register ``blog.js`` as a hosted script (with its SRI hash) and
  add it to the site's or a page's footer, keeping the scripts already there.

Nothing here publishes: the client refuses publish endpoints, so the site
changes only when someone publishes it in Webflow. Before ``push`` or
``script`` sends anything, they fetch the assets from the pinned
``--asset-ref`` and check they match the local build, so a post never points
at missing or different files.

The token comes from ``$WEBFLOW_TOKEN`` or ``--token-file`` and is never
printed. Scopes: ``sites:read`` and ``cms:read`` for ``discover`` and dry runs,
plus ``cms:write`` for ``push --apply``, ``pages:read`` for template pages and
``custom_code:read``/``custom_code:write`` for ``script``.

Usage (from the repository root, standard library only)::

    python assets/blog/webflow_api.py discover
    python assets/blog/webflow_api.py push --collection ID --asset-ref v0.2.0
    python assets/blog/webflow_api.py script --site ID --asset-ref v0.2.0
"""

import argparse
import base64
import dataclasses
import difflib
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import build
import webflow

API = "https://api.webflow.com/v2/"
TOKEN_ENV = "WEBFLOW_TOKEN"  # noqa: S105  # nosec B105 (a variable name)
# Proposed slugs; the posts' URLs are --post-base followed by these.
SLUGS = {
    "index.html": "does-a-decision-model-know-when-it-is-guessing",
    "part-1.html": "does-a-decision-model-know-when-it-is-guessing-part-1",
    "part-2.html": "does-a-decision-model-know-when-it-is-guessing-part-2",
}
DEFAULT_PAGES = ("part-1.html", "part-2.html")
SCRIPT_NAME = "BeyondAnswerConfidenceBlog"
PAGE_SIZE = 100
RETRIES = 3
PUBLISHING = re.compile(r"(?:^|/)publish(?:$|/)|/items/live(?:$|/)")
SEMVER = re.compile(r"v?(\d+\.\d+\.\d+)")

# (method, url, headers, body) -> (status, response body)
Send = Callable[[str, str, dict[str, str], bytes | None], tuple[int, bytes]]
Fetch = Callable[[str], bytes]


class ApiError(Exception):
    """A refused or failed API request, or a precondition that stops a command."""


def _https(url: str) -> None:
    if not url.startswith("https://"):
        raise ApiError(f"refusing non-HTTPS URL {url}")


def urllib_send(
    method: str, url: str, headers: dict[str, str], body: bytes | None
) -> tuple[int, bytes]:
    """Send one HTTP request with urllib.

    Args:
        method: HTTP method.
        url: Absolute HTTPS URL.
        headers: Request headers.
        body: Request body, if any.

    Returns:
        Status code and response body (also for error statuses).
    """
    _https(url)
    req = urllib.request.Request(url, data=body, headers=headers, method=method)  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310  # nosec B310
            return int(resp.status), bytes(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def urllib_fetch(url: str) -> bytes:
    """Download a public asset.

    Args:
        url: Absolute HTTPS URL.

    Returns:
        The response body.

    Raises:
        ApiError: On a non-200 response.
    """
    status, body = urllib_send("GET", url, {"User-Agent": "webflow_api.py"}, None)
    if status != 200:
        raise ApiError(f"{url}: HTTP {status}")
    return body


class Client:
    """Webflow Data API v2 client that writes only when allowed to.

    Args:
        token: API token (kept private).
        send: HTTP transport (default: urllib).
        writes: Whether non-GET requests may be sent.
        sleep: Called with seconds to wait on HTTP 429.
    """

    def __init__(
        self,
        token: str,
        send: Send = urllib_send,
        writes: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Set up the client (see the class docstring)."""
        self._token = token
        self._send = send
        self.writes = writes
        self._sleep = sleep

    def __repr__(self) -> str:
        """Describe the client without its token."""
        return f"Client(writes={self.writes})"

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        params: dict[str, str | int] | None = None,
    ) -> Any:
        """Send a request and decode the JSON response.

        Args:
            method: HTTP method.
            path: Path below ``API``.
            body: JSON body, if any.
            params: Query parameters.

        Returns:
            The decoded response (None for an empty body).

        Raises:
            ApiError: For publish endpoints, writes in a dry run, and failed
                requests.
        """
        if PUBLISHING.search(path):
            raise ApiError(f"refusing to publish ({path}): publish in Webflow")
        if method != "GET" and not self.writes:
            raise ApiError(f"dry run: refusing {method} {path}")
        url = API + path + (f"?{urllib.parse.urlencode(params)}" if params else "")
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        for attempt in range(RETRIES + 1):
            status, raw = self._send(method, url, headers, data)
            if status != 429 or attempt == RETRIES:
                break
            self._sleep(60)
        if not 200 <= status < 300:
            detail = raw.decode(errors="replace")[:300]
            raise ApiError(f"{method} {path}: HTTP {status}: {detail}")
        return json.loads(raw) if raw else None

    def get(self, path: str, **params: str | int) -> Any:
        """GET a path (see ``request``)."""
        return self.request("GET", path, params=params or None)

    def pages(self, path: str, key: str) -> list[dict[str, Any]]:
        """GET every page of a paginated list.

        Args:
            path: List endpoint.
            key: Name of the list in the response (``items`` etc.).

        Returns:
            All entries.
        """
        out: list[dict[str, Any]] = []
        while True:
            resp = self.get(path, offset=len(out), limit=PAGE_SIZE)
            batch = resp.get(key, [])
            out += batch
            total = resp.get("pagination", {}).get("total", len(out))
            if not batch or len(out) >= total:
                return out


# -- discover -----------------------------------------------------------------


def discover(client: Client, site: str | None = None) -> list[str]:
    """Describe the sites, collections, fields and template pages.

    Args:
        client: API client.
        site: Only this site id (default: every site the token sees).

    Returns:
        Report lines.
    """
    sites = [client.get(f"sites/{site}")] if site else client.get("sites")["sites"]
    lines = []
    for s in sites:
        domains = ", ".join(d.get("url", "") for d in s.get("customDomains", []))
        lines.append(
            f"site {s['id']}  {s.get('displayName', '')}  "
            f"({s.get('shortName', '')}.webflow.io{', ' + domains if domains else ''})"
        )
        for c in client.get(f"sites/{s['id']}/collections")["collections"]:
            detail = client.get(f"collections/{c['id']}")
            lines.append(
                f"  collection {c['id']}  {c.get('displayName', '')}  "
                f"/{c.get('slug', '')}/"
            )
            for f in detail.get("fields", []):
                req = " required" if f.get("isRequired") else ""
                lines.append(
                    f"    field {f['slug']:<28} {f['type']:<14}"
                    f" {f.get('displayName', '')}{req}"
                )
        try:
            pages = client.pages(f"sites/{s['id']}/pages", "pages")
        except ApiError as err:
            lines.append(f"  pages: {err}")
            continue
        lines.extend(
            f"  template page {p['id']}  {p.get('title', '')}  "
            f"(collection {p['collectionId']})"
            for p in pages
            if p.get("collectionId")
        )
    return lines


# -- assets -------------------------------------------------------------------


def asset_base(ref: str) -> str:
    """URL of ``webflow/`` at a pinned git ref.

    Args:
        ref: Git tag or commit.

    Returns:
        The jsDelivr URL, ending in ``/``.

    Raises:
        ApiError: For a branch name such as ``main``, which can move.
    """
    if ref in {"main", "master", "HEAD"}:
        raise ApiError(f"--asset-ref {ref} can move under a live post; use a tag")
    return webflow.ASSET_BASE.format(ref=ref)


def check_assets(
    base: str, files: dict[str, str], fetch: Fetch = urllib_fetch
) -> list[str]:
    """Compare the served assets with the local build.

    Args:
        base: URL of ``webflow/``.
        files: The local build (see ``webflow.build_files``).
        fetch: Downloads a URL.

    Returns:
        One problem per asset that is missing or differs (empty if all match).
    """
    problems = []
    for name in sorted(n for n in files if not n.endswith(".html")):
        try:
            served = fetch(base + name)
        except (ApiError, OSError) as err:
            problems.append(f"{name}: {err}")
            continue
        if served != files[name].encode():
            problems.append(f"{name}: served file differs from the local build")
    return problems


def _require_assets(base: str, files: dict[str, str], fetch: Fetch) -> None:
    problems = check_assets(base, files, fetch)
    if problems:
        raise ApiError(
            "assets at the pinned ref do not match the build:\n  "
            + "\n  ".join(problems)
            + "\n(tag a commit with the current webflow/ files and push the tag)"
        )


# -- push ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Plan:
    """What ``push`` would do for one page.

    Attributes:
        page: Page file name.
        slug: Item slug.
        fields: Field data to send.
        item_id: Existing item, or None to create one.
        changed: Names of fields that differ from the existing item.
        diff: Unified diff of changed text fields.
    """

    page: str
    slug: str
    fields: dict[str, str]
    item_id: str | None
    changed: tuple[str, ...]
    diff: tuple[str, ...]


def _field_types(client: Client, collection: str) -> dict[str, str]:
    detail = client.get(f"collections/{collection}")
    return {f["slug"]: f["type"] for f in detail.get("fields", [])}


def plan_push(
    client: Client,
    collection: str,
    rich: dict[str, str],
    pages: Iterable[str],
    slugs: dict[str, str],
    body_field: str,
    summary_field: str | None,
    update_meta: bool = False,
) -> list[Plan]:
    """Work out the create/update for each page.

    Args:
        client: API client (only reads).
        collection: Collection id.
        rich: Rich-text body by page file name.
        pages: Page file names to push.
        slugs: Item slug by page file name.
        body_field: Slug of the rich-text field for the body.
        summary_field: Slug of a plain-text field for the page description.
        update_meta: Also update the title and summary of existing items
            (by default they are set only when an item is created, so edits
            made in Webflow stay).

    Returns:
        One plan per page.

    Raises:
        ApiError: If a field is missing or the body field is not rich text.
    """
    types = _field_types(client, collection)
    wanted = {"name": "PlainText", "slug": "PlainText", body_field: "RichText"}
    if summary_field:
        wanted[summary_field] = "PlainText"
    for field, kind in wanted.items():
        if field not in types:
            raise ApiError(f"collection has no field {field!r}: {sorted(types)}")
        if kind == "RichText" and types[field] != kind:
            raise ApiError(f"field {field!r} is {types[field]}, not RichText")
    existing = {
        it.get("fieldData", {}).get("slug"): it
        for it in client.pages(f"collections/{collection}/items", "items")
    }
    by_name = {p.filename: p for p in build.PAGES}
    plans = []
    for page in pages:
        fields = {
            "name": by_name[page].title,
            "slug": slugs[page],
            body_field: rich[page],
        }
        if summary_field:
            fields[summary_field] = by_name[page].description
        item = existing.get(slugs[page])
        old = item.get("fieldData", {}) if item else {}
        compared = fields if not item or update_meta else {body_field: rich[page]}
        changed = tuple(k for k, v in compared.items() if old.get(k) != v)
        diff: list[str] = []
        for k in changed if item else ():
            diff += difflib.unified_diff(
                str(old.get(k, "")).splitlines(),
                fields[k].splitlines(),
                f"{slugs[page]}:{k} (Webflow)",
                f"{slugs[page]}:{k} (build)",
                lineterm="",
            )
        plans.append(
            Plan(
                page,
                slugs[page],
                fields,
                item["id"] if item else None,
                changed,
                tuple(diff),
            )
        )
    return plans


def apply_push(client: Client, collection: str, plans: list[Plan]) -> list[str]:
    """Create or update the items, then read them back.

    Args:
        client: API client that may write.
        collection: Collection id.
        plans: From ``plan_push``.

    Returns:
        Report lines.

    Raises:
        ApiError: If an item was not stored as sent.
    """
    lines = []
    for p in plans:
        if p.item_id and not p.changed:
            continue
        if p.item_id:
            sent = {k: p.fields[k] for k in p.changed}
            item = client.request(
                "PATCH",
                f"collections/{collection}/items/{p.item_id}",
                {"fieldData": sent},
            )
        else:
            sent = p.fields
            item = client.request(
                "POST",
                f"collections/{collection}/items",
                {"isDraft": True, "isArchived": False, "fieldData": sent},
            )
        back = client.get(f"collections/{collection}/items/{item['id']}")["fieldData"]
        bad = sorted(k for k, v in sent.items() if back.get(k) != v)
        if bad:
            raise ApiError(f"{p.slug}: stored differently from what was sent: {bad}")
        verb = "updated" if p.item_id else "created draft"
        lines.append(f"{verb} {p.slug} ({item['id']}), read back identical")
    return lines


def describe(plans: list[Plan], show_diff: bool) -> list[str]:
    """Report lines for a dry run.

    Args:
        plans: From ``plan_push``.
        show_diff: Include the unified diffs.

    Returns:
        Report lines.
    """
    lines = []
    for p in plans:
        size = f"{max(map(len, p.fields.values())):,} characters"
        if not p.item_id:
            lines.append(f"would create draft {p.slug} from {p.page} ({size})")
        elif not p.changed:
            lines.append(f"unchanged {p.slug} ({p.item_id})")
        else:
            lines.append(
                f"would update {p.slug} ({p.item_id}): {', '.join(p.changed)} ({size})"
            )
            if show_diff:
                lines += p.diff
    return lines


# -- script -------------------------------------------------------------------


def sri(text: str) -> str:
    """Subresource-integrity hash of a file.

    Args:
        text: File contents.

    Returns:
        ``sha384-<base64>``.
    """
    digest = hashlib.sha384(text.encode()).digest()
    return "sha384-" + base64.b64encode(digest).decode()


def plan_script(
    client: Client,
    site: str,
    target: str,
    url: str,
    integrity: str,
    version: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    """Work out the script registration and the new custom-code list.

    Args:
        client: API client (only reads).
        site: Site id (scripts are registered per site).
        target: ``sites/<id>`` or ``pages/<id>``, where the script is added.
        url: Hosted URL of ``blog.js``.
        integrity: Its SRI hash.
        version: Script version (semantic version).

    Returns:
        The registration to create (None if this version is registered),
        the target's full script list afterwards, and report lines.

    Raises:
        ApiError: If this version is registered with another URL or hash.
    """
    registered = client.get(f"sites/{site}/registered_scripts").get(
        "registeredScripts", []
    )
    ours = [s for s in registered if s.get("displayName") == SCRIPT_NAME]
    same = [s for s in ours if s.get("version") == version]
    lines = []
    new = None
    if same:
        s = same[0]
        if s.get("hostedLocation") != url or s.get("integrityHash") != integrity:
            raise ApiError(
                f"{SCRIPT_NAME} {version} is registered with another URL or hash; "
                "use a new version"
            )
        script_id = s["id"]
        lines.append(f"{SCRIPT_NAME} {version} is registered ({script_id})")
    else:
        new = {
            "displayName": SCRIPT_NAME,
            "hostedLocation": url,
            "integrityHash": integrity,
            "version": version,
            "canCopy": False,
        }
        script_id = ours[0]["id"] if ours else None
        lines.append(f"would register {SCRIPT_NAME} {version} from {url}")
    current = client.get(f"{target}/custom_code").get("scripts", [])
    ids = {s["id"] for s in ours}
    kept = [s for s in current if s.get("id") not in ids]
    if len(kept) != len(current):
        lines.append(f"would replace the older {SCRIPT_NAME} entry on {target}")
    lines.append(
        f"would set {target} custom code: {len(kept)} other script(s) kept, "
        f"{SCRIPT_NAME} {version} in the footer"
    )
    entry = {"id": script_id, "location": "footer", "version": version}
    return new, [*kept, entry], lines


def apply_script(
    client: Client,
    site: str,
    target: str,
    new: dict[str, Any] | None,
    scripts: list[dict[str, Any]],
) -> list[str]:
    """Register the script if needed and set the target's custom code.

    Args:
        client: API client that may write.
        site: Site id.
        target: ``sites/<id>`` or ``pages/<id>``.
        new: Registration to create, or None.
        scripts: From ``plan_script``.

    Returns:
        Report lines.
    """
    lines = []
    if new is not None:
        made = client.request("POST", f"sites/{site}/registered_scripts/hosted", new)
        scripts[-1]["id"] = made["id"]
        lines.append(f"registered {SCRIPT_NAME} {new['version']} ({made['id']})")
    client.request("PUT", f"{target}/custom_code", {"scripts": scripts})
    lines.append(
        f"set {target} custom code ({len(scripts)} script(s)); publish to apply"
    )
    return lines


# -- command line -------------------------------------------------------------


def _token(path: Path | None) -> str:
    token = (
        path.read_text(encoding="utf-8").strip()
        if path
        else os.environ.get(TOKEN_ENV, "")
    )
    if not token:
        raise ApiError(f"no token: set ${TOKEN_ENV} or pass --token-file")
    return token


def _slugs(overrides: list[str]) -> dict[str, str]:
    slugs = dict(SLUGS)
    for o in overrides:
        page, sep, slug = o.partition("=")
        if not sep or page not in slugs or not re.fullmatch(r"[a-z0-9-]+", slug):
            raise ApiError(f"--slug wants PAGE=slug with PAGE one of {sorted(slugs)}")
        slugs[page] = slug
    return slugs


def _version(ref: str, version: str | None) -> str:
    if version:
        return version
    m = SEMVER.fullmatch(ref)
    if not m:
        raise ApiError(f"--asset-ref {ref} is not a version tag; pass --version")
    return m.group(1)


def _push(args: argparse.Namespace, client: Client, fetch: Fetch) -> list[str]:
    base = asset_base(args.asset_ref)
    slugs = _slugs(args.slug)
    urls = {page: args.post_base + slug for page, slug in slugs.items()}
    files, rich = webflow.build_files(base, urls)
    if args.no_asset_check:
        if args.apply:
            raise ApiError("--no-asset-check is for dry runs only")
    else:
        _require_assets(base, files, fetch)
    plans = plan_push(
        client,
        args.collection,
        rich,
        args.page or DEFAULT_PAGES,
        slugs,
        args.body_field,
        args.summary_field,
        args.update_meta,
    )
    if not args.apply:
        return [*describe(plans, args.diff), "dry run: nothing sent (--apply sends)"]
    return apply_push(client, args.collection, plans)


def _script(args: argparse.Namespace, client: Client, fetch: Fetch) -> list[str]:
    base = asset_base(args.asset_ref)
    files, _ = webflow.build_files(base)
    _require_assets(base, files, fetch)
    target = f"pages/{args.page}" if args.page else f"sites/{args.site}"
    new, scripts, lines = plan_script(
        client,
        args.site,
        target,
        base + "blog.js",
        sri(files["blog.js"]),
        _version(args.asset_ref, args.version),
    )
    if not args.apply:
        return [*lines, "dry run: nothing sent (--apply sends)"]
    return apply_script(client, args.site, target, new, scripts)


def parser() -> argparse.ArgumentParser:
    """The command-line parser.

    Returns:
        Parser with the ``discover``, ``push`` and ``script`` commands.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--token-file", type=Path, help=f"file with the token (default ${TOKEN_ENV})"
    )
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser(
        "discover", help="list sites, collections, fields, template pages"
    )
    d.add_argument("--site", help="only this site id")
    p = sub.add_parser("push", help="create or update the posts' CMS items")
    p.add_argument("--collection", required=True, help="blog collection id")
    p.add_argument(
        "--asset-ref", required=True, help="git tag the assets are served from"
    )
    p.add_argument(
        "--page",
        action="append",
        choices=sorted(SLUGS),
        help="page to push (repeatable; default: both parts)",
    )
    p.add_argument(
        "--slug",
        action="append",
        default=[],
        metavar="PAGE=SLUG",
        help="override a slug",
    )
    p.add_argument(
        "--post-base",
        default="/blog/",
        help="URL path the slugs follow (default /blog/)",
    )
    p.add_argument("--body-field", default="body", help="rich-text field for the body")
    p.add_argument("--summary-field", help="plain-text field for the page description")
    p.add_argument(
        "--update-meta",
        action="store_true",
        help="also overwrite the title and summary of existing items",
    )
    p.add_argument(
        "--diff", action="store_true", help="show the diff of changed fields"
    )
    p.add_argument(
        "--no-asset-check",
        action="store_true",
        help="skip the asset check (dry run only)",
    )
    p.add_argument("--apply", action="store_true", help="send the changes")
    s = sub.add_parser("script", help="register blog.js and add it to the footer")
    s.add_argument("--site", required=True, help="site id")
    s.add_argument("--page", help="add it to this page instead of the whole site")
    s.add_argument(
        "--asset-ref", required=True, help="git tag the assets are served from"
    )
    s.add_argument("--version", help="script version (default: from the tag)")
    s.add_argument("--apply", action="store_true", help="send the changes")
    return ap


def main(
    argv: list[str] | None = None,
    send: Send = urllib_send,
    fetch: Fetch = urllib_fetch,
) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).
        send: HTTP transport for the API.
        fetch: Downloads assets.

    Returns:
        Process exit code.
    """
    args = parser().parse_args(argv)
    try:
        client = Client(
            _token(args.token_file), send, writes=getattr(args, "apply", False)
        )
        if args.command == "discover":
            lines = discover(client, args.site)
        elif args.command == "push":
            lines = _push(args, client, fetch)
        else:
            lines = _script(args, client, fetch)
    except (ApiError, build.BuildError) as err:
        sys.stderr.write(f"webflow_api: {err}\n")
        return 1
    sys.stdout.write("".join(f"{line}\n" for line in lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
