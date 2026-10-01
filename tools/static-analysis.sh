#!/usr/bin/env bash
# Type-check with mypy (strict, config in pyproject.toml) and scan the package
# with bandit.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Running mypy..."
uv run mypy

echo "Running bandit..."
uv run bandit -c pyproject.toml -q -r src
