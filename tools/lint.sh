#!/usr/bin/env bash
# Lint and format the tree with ruff. Pass --check to verify without writing
# (this is what CI does).
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--check" ]]; then
  echo "Running ruff linter (check only)..."
  uv run ruff check --no-fix
  echo "Running ruff formatter (check only)..."
  uv run ruff format --check
else
  echo "Running ruff linter..."
  uv run ruff check --exit-non-zero-on-fix
  echo "Running ruff formatter..."
  uv run ruff format
fi
