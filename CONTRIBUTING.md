# Contributing

Thanks for your interest in improving `beyond-answer-confidence`.

## Development setup

You need [uv](https://docs.astral.sh/uv/) and git. uv installs the right
Python (3.13) and all dependencies from the committed `uv.lock`.

```bash
git clone <your fork>
cd beyond-answer-confidence
uv sync                      # project + dev and lint groups
uv run pre-commit install    # installs the pre-commit and pre-push hooks
```

Optional local-model features need the `models` extra: `uv sync --extra models`.

## Quality gates

Every change must pass the same checks CI runs:

| Check | Command |
|---|---|
| Lint and format (ruff) | `tools/lint.sh` (`tools/lint.sh --check` to verify only) |
| Types (mypy, strict) and bandit | `tools/static-analysis.sh` |
| Tests with coverage floor | `uv run pytest` |
| Secret scan (detect-secrets, gitleaks) | `tools/secret-scan.sh` |
| Leak audit | `tools/leak-audit.sh` |
| All hooks | `uv run pre-commit run --all-files` |

The leak audit runs generic checks (credentials, e-mail addresses, home
directory paths, `.env` files, notebook outputs, large files, to-do markers).
Maintainers can add their own patterns without committing them: put one
regular expression per line in a file outside the repository and point
`LEAK_AUDIT_TERMS_FILE` at it.

Conventions:

- Type annotations everywhere (`mypy --strict`).
- Google-style docstrings on public functions and classes.
- `logging`, not `print`, in library code.
- Pure functions for metrics and builders; file and network access stay in the
  data, cache and CLI layers. Randomness goes through a passed
  `numpy.random.Generator` or seed.
- Add dependencies with `uv add <package>` (never by editing the dependency
  lists by hand), and commit the updated `uv.lock`.

## Tests run offline

The test suite never touches the network or a paid API. Use the deterministic
fake backend (`beyond_answer_confidence.backends.fake.FakeBackend`) and small synthetic
inputs. Mark long-running tests with `@pytest.mark.slow`.

## What must not be committed

- API keys, tokens or `.env` files of any kind.
- Dataset text or samples, including in test fixtures (use synthetic data).
- Raw API responses, response caches, or run outputs (`data/`, `outputs/`).
- Model checkpoints or other large binaries.
- Notebook outputs (`nbstripout` removes them on commit; do not disable it).

The pre-commit hooks and CI check for most of these, but they are a safety
net, not a substitute for care.

## Pull requests

- Keep each pull request focused on one change and describe why it is needed.
- Add or update tests for behaviour changes.
- Add a line to the `Unreleased` section of `CHANGELOG.md` for user-visible
  changes.
- By contributing you agree that your contribution is licensed under the
  Apache License 2.0, and that you follow the [Code of Conduct](CODE_OF_CONDUCT.md).

To report a security problem, see [SECURITY.md](SECURITY.md) instead of opening
an issue.
