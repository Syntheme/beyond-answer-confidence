#!/usr/bin/env bash
# Full secret sweep:
#   - detect-secrets over the working tree, against .secrets.baseline, and a
#     check that every baseline entry has been audited as a false positive;
#   - gitleaks over the working tree and the full git history.
# Exits non-zero on any finding. Findings are printed redacted.
set -euo pipefail

cd "$(dirname "$0")/.."

status=0

# Tracked files plus untracked files that are not ignored (works before the
# first commit), minus the lock file.
mapfile -t files < <(git ls-files --cached --others --exclude-standard \
  | grep -v -E '^(uv\.lock)$' || true)

echo "=== detect-secrets: working tree vs .secrets.baseline ==="
if ((${#files[@]})); then
  uv run --no-sync detect-secrets-hook --baseline .secrets.baseline "${files[@]}" || status=1
fi
uv run --no-sync python - <<'PY' || status=1
import json
import sys

baseline = json.load(open(".secrets.baseline", encoding="utf-8"))
bad = [
    f"{name}:{r['line_number']} ({r['type']})"
    for name, results in baseline["results"].items()
    for r in results
    if r.get("is_secret") is not False
]
if bad:
    print("Baseline entries not audited as false positives:")
    print("\n".join(f"  {b}" for b in bad))
    sys.exit(1)
print(f"baseline: {sum(map(len, baseline['results'].values()))} entries, all audited")
PY

echo
echo "=== gitleaks ==="
gitleaks_bin="$(command -v gitleaks || true)"
if [[ -z "$gitleaks_bin" ]]; then
  # Fall back to the binary pre-commit built for the gitleaks hook.
  cache="${PRE_COMMIT_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/pre-commit}"
  gitleaks_bin="$(find "$cache" -type f -name gitleaks -path '*golangenv*' \
    -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n1 | cut -d' ' -f2- || true)"
fi
if [[ -z "$gitleaks_bin" ]]; then
  echo "gitleaks not found. Install it, or run 'uv run pre-commit install-hooks'."
  exit 1
fi

echo "--- working tree ---"
tree="$(mktemp -d)"
trap 'rm -rf "$tree"' EXIT
if ((${#files[@]})); then
  cp --parents -- "${files[@]}" "$tree"/ 2>/dev/null || true
fi
"$gitleaks_bin" dir --redact --no-banner "$tree" || status=1

if git rev-parse --verify -q HEAD >/dev/null; then
  echo "--- full git history ---"
  "$gitleaks_bin" git --redact --no-banner --log-opts="--all" . || status=1
else
  echo "(no commits yet; history scan skipped)"
fi

exit "$status"
