#!/usr/bin/env bash
# Leak audit: look for material that must never reach a public repository.
# Scans
#   1. the working tree (tracked files plus untracked files that are not
#      ignored), so it works before the first commit;
#   2. every commit reachable from any ref: added and removed diff lines, file
#      paths, commit messages and author/committer identities.
#
# Built-in checks are generic: credential and key patterns, e-mail addresses
# (other than GitHub noreply addresses), absolute home-directory paths, .env
# files, key material, notebook outputs, .DS_Store files, large binaries,
# data/output directories, model checkpoints and T[O]DO/FI[X]ME markers.
#
# Extra, project-specific patterns can be kept outside the repository and
# loaded from the file named by the LEAK_AUDIT_TERMS_FILE environment
# variable: one regular expression per line (Python `re` syntax, a superset of
# POSIX ERE; start a pattern with (?i) for case-insensitive matching), blank
# lines and lines starting with # ignored. A line may be "label<TAB>pattern"
# to give its findings a label.
#
# A line containing the marker "leak-audit: allow" is skipped.
#
# Prints file:line (or commit:path:line) for each hit and exits 1 if there is
# any hit, 0 otherwise (2 on a usage or terms-file error).
#
# Usage: [LEAK_AUDIT_TERMS_FILE=FILE] tools/leak-audit.sh [--tree-only | --history-only]
#
# Some built-in patterns are written with a bracketed character (e.g.
# "w[o]rd") so that this script does not match itself.
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v python3 >/dev/null 2>&1; then
  PYTHON=(python3)
else
  PYTHON=(uv run --no-sync python)
fi

exec "${PYTHON[@]}" - "$@" <<'PY'
"""Leak audit implementation (see tools/leak-audit.sh)."""

import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator

I = re.IGNORECASE

# Built-in generic patterns: (label, compiled pattern).
TEXT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("local path", re.compile(r"/work[s]paces\b")),
    ("local path", re.compile(r"/h[o]me/[A-Za-z0-9._-]+")),
    ("local path", re.compile(r"/U[s]ers/[A-Za-z0-9._-]+/")),
    ("local path", re.compile(r"(?i)\b[A-Z]:\\U[s]ers\\")),
    ("marker", re.compile(r"\b(?:T[O]DO|FI[X]ME|X[X]X)\b")),
    ("credential", re.compile(r"\bs[k]-[A-Za-z0-9_-]")),
    ("credential", re.compile(r"A[I]za[0-9A-Za-z_-]")),
    ("credential", re.compile(r"\bbe[a]rer\s+[A-Za-z0-9._~+/-]", I)),
    ("credential", re.compile(r"\b(?:A[K]IA|A[S]IA)[0-9A-Z]{16}\b")),
    ("credential", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("credential", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("credential", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "credential",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|passw(?:or)?d|token)\b\s*[:=]\s*"
            r"[\"'][A-Za-z0-9_\-./+]{16,}[\"']"
        ),
    ),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")),
]

# Email addresses that are fine to appear anywhere: GitHub noreply addresses
# and the paper authors' published contact addresses (assets/paper/main.tex).
ALLOWED_EMAIL = re.compile(
    r"(?:[A-Za-z0-9._%+\[\]-]+@users\.noreply\.github\.com"
    r"|noreply@github\.com|git@github\.com"
    r"|(?:sharath|davor|jan)@synthpop\.ai)$",
)

# Paths that must never exist.
BAD_PATHS: list[tuple[str, re.Pattern[str]]] = [
    ("env file", re.compile(r"(?:^|/)\.env(?:\.(?!example$)[^/]*)?$|(?<!\.example)\.env$")),
    ("macOS metadata", re.compile(r"(?:^|/)\.DS_Store$")),
    ("key material", re.compile(r"\.(?:pem|key|p12|pfx|jks|keystore)$|(?:^|/)id_(?:rsa|ecdsa|ed25519)")),
    ("data directory", re.compile(r"^(?:data|outputs)/")),
    ("checkpoint", re.compile(r"\.(?:ckpt|pt|pth|safetensors)$")),
]

# Tracked files larger than this are reported (binaries and data dumps belong
# in releases or external storage, not in git).
MAX_FILE_BYTES = 1_000_000
LARGE_FILE_EXEMPT = {"uv.lock"}

# Files whose text is not scanned: verbatim third-party text. Add others here
# only if they cause false positives. Path checks still apply to them.
TEXT_EXCLUDE = {"CODE_OF_CONDUCT.md"}

# A line containing this marker is not reported. Use it sparingly, only for
# identifiers that cannot change (e.g. keys that must stay byte-compatible).
ALLOW_MARKER = "leak-audit: allow"

TERMS_ENV = "LEAK_AUDIT_TERMS_FILE"

hits = 0


def load_terms() -> list[tuple[str, re.Pattern[str]]]:
    """Load extra patterns from the file named by $LEAK_AUDIT_TERMS_FILE.

    Returns:
        (label, pattern) pairs; empty if the variable is unset or empty.
    """
    path = os.environ.get(TERMS_ENV, "")
    if not path:
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError as err:
        sys.exit(f"leak audit: cannot read {TERMS_ENV} file: {err.strerror}")
    terms = []
    for no, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        label, sep, pattern = raw.strip().partition("\t")
        if not sep:
            label, pattern = "project term", line
        try:
            terms.append((label.strip(), re.compile(pattern.strip())))
        except re.error as err:
            sys.exit(f"leak audit: bad pattern on line {no} of the terms file: {err}")
    return terms


def report(where: str, label: str, text: str) -> None:
    """Print one finding."""
    global hits
    hits += 1
    snippet = text.strip()
    if len(snippet) > 160:
        snippet = snippet[:157] + "..."
    print(f"{where}: [{label}] {snippet}")


def git(*args: str) -> str:
    """Run a git command and return stdout (empty string on failure)."""
    res = subprocess.run(
        ["git", *args], capture_output=True, text=True, errors="replace", check=False
    )
    return res.stdout if res.returncode == 0 else ""


def scan_line(where: str, line: str) -> None:
    """Report every pattern that matches the line."""
    if ALLOW_MARKER in line:
        return
    for label, pat in TEXT_PATTERNS:
        if label == "email":
            if any(not ALLOWED_EMAIL.match(m.group()) for m in pat.finditer(line)):
                report(where, label, line)
        elif pat.search(line):
            report(where, label, line)


def scan_size(where: str, size: int, path: str) -> None:
    """Report files above the size limit."""
    if size > MAX_FILE_BYTES and path not in LARGE_FILE_EXEMPT:
        report(where, "large file", f"{path} ({size:,} bytes)")


def scan_path(where: str, path: str) -> None:
    """Report forbidden file paths."""
    for label, pat in BAD_PATHS:
        if pat.search(path):
            report(where, label, path)


def notebook_has_outputs(text: str) -> bool:
    """Return True if a notebook JSON carries outputs or execution counts."""
    try:
        nb = json.loads(text)
    except ValueError:
        return False
    for cell in nb.get("cells", []):
        if cell.get("outputs") or cell.get("execution_count") is not None:
            return True
    return False


def scan_tree() -> None:
    """Scan the working tree (tracked + untracked, not ignored)."""
    listing = git("ls-files", "-z", "--cached", "--others", "--exclude-standard")
    for path in sorted({p for p in listing.split("\0") if p}):
        scan_path(path, path)
        if path in TEXT_EXCLUDE:
            continue
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        scan_size(path, len(data), path)
        if b"\0" in data[:8192]:
            continue  # binary
        text = data.decode("utf-8", errors="replace")
        if path.endswith(".ipynb") and notebook_has_outputs(text):
            report(path, "notebook outputs", "cells carry outputs")
        for no, line in enumerate(text.splitlines(), 1):
            scan_line(f"{path}:{no}", line)


def history_lines() -> Iterator[tuple[str, str, str]]:
    """Yield (commit, path, line) for every added/removed line in history."""
    out = git("log", "--all", "-p", "--no-color", "--no-ext-diff", "--format=commit %H")
    commit, path = "?", "?"
    for line in out.splitlines():
        if line.startswith("commit ") and len(line) == 47:
            commit = line[7:14]
        elif line.startswith("+++ ") or line.startswith("--- "):
            p = line[4:]
            if p != "/dev/null":
                path = p[2:] if p[:2] in ("a/", "b/") else p
        elif line[:1] in "+-" and not line.startswith(("+++", "---")):
            yield commit, path, line[1:]


def scan_history() -> None:
    """Scan diffs, paths, messages and identities of every commit."""
    if not git("rev-parse", "--verify", "-q", "HEAD").strip():
        print("(no commits yet; history scan skipped)")
        return
    for commit, path, line in history_lines():
        if path in TEXT_EXCLUDE:
            continue
        if path.endswith(".ipynb") and '"output_type"' in line:
            report(f"{commit}:{path}", "notebook outputs", line)
        scan_line(f"{commit}:{path}", line)
    for entry in git("log", "--all", "--name-only", "--format=@%h").splitlines():
        if entry.startswith("@"):
            commit = entry[1:]
        elif entry:
            scan_path(f"{commit}:{entry}", entry)
    for commit in git("rev-list", "--all").split():
        for rec in git("ls-tree", "-r", "-l", "-z", commit).split("\0"):
            meta, _, path = rec.partition("\t")
            fields = meta.split()
            if len(fields) == 4 and fields[1] == "blob" and fields[3].isdigit():
                scan_size(f"{commit[:7]}:{path}", int(fields[3]), path)
    for rec in git("log", "--all", "--format=%h%x00%B%x01").split("\x01"):
        if "\0" not in rec:
            continue
        commit, body = rec.strip("\n").split("\0", 1)
        for no, line in enumerate(body.splitlines(), 1):
            scan_line(f"{commit}:<message>:{no}", line)
    for rec in git("log", "--all", "--format=%h%x00%an <%ae>%x00%cn <%ce>").splitlines():
        commit, author, committer = rec.split("\0")
        for role, ident in (("author", author), ("committer", committer)):
            email = ident.rsplit("<", 1)[-1].rstrip(">")
            if not ALLOWED_EMAIL.match(email):
                report(f"{commit}:<{role}>", "identity", "non-noreply email: " + ident)


def main() -> int:
    """Run the audit and return the exit status."""
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("", "--tree-only", "--history-only"):
        print(__doc__, "\nusage: leak-audit.sh [--tree-only | --history-only]")
        return 2
    extra = load_terms()
    TEXT_PATTERNS.extend(extra)
    if extra:
        print(f"(+{len(extra)} pattern(s) from ${TERMS_ENV})")
    if mode != "--history-only":
        print("== working tree ==")
        scan_tree()
    if mode != "--tree-only":
        print("== git history ==")
        scan_history()
    print(f"leak audit: {hits} hit(s)")
    return 1 if hits else 0


sys.exit(main())
PY
