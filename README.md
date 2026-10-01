# beyond-answer-confidence

Probe whether a decision API's probabilities track what it knows.

`beyond-answer-confidence` sends controlled multiple-choice and yes/no questions to a
decision API that returns a probability for every option, and measures how
those probabilities behave. The experiments cover:

- **calibration** on public question-answering and multiple-choice
  benchmarks, and how it changes with the number of options;
- **knowledge dial**: intent classification with opaque codes and a
  controlled amount of knowledge about them (none, examples, names);
- **knowledge boundary**: popular versus obscure entities, fabricated
  entities, and news questions dated around a knowledge cutoff;
- **evidence sufficiency**: questions with no, partial or full supporting
  evidence, and clues revealed one at a time;
- **out-of-scope** detection, **synthetic worlds** with known ideal answers,
  **stated odds** (dice, coins, cards), re-worded follow-up questions and
  local comparators (a deep ensemble and a zero-shot classifier).

Each experiment rebuilds its requests deterministically from pinned dataset
revisions, answers them through a local cache, scores the answers and writes
per-item rows and a summary of effect sizes with bootstrap intervals.

## Install

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```sh
git clone <this repository> && cd beyond-answer-confidence
uv sync                    # core
uv sync --extra models     # + torch/transformers (distractors, deep ensemble)
uv run beyond-answer-confidence list
```

## Quick start (offline, no key)

```sh
uv run python examples/quickstart.py
```

This builds a few toy items, answers them with the deterministic
`FakeBackend` through a temporary cache (`runner.run_items`), and prints the
scored rows. No network access and no API key are needed.

## Running an experiment

```sh
uv run beyond-answer-confidence run synthetic_worlds            # offline (default)
uv run beyond-answer-confidence estimate synthetic_worlds       # uncached requests, token estimate
uv run beyond-answer-confidence run synthetic_worlds --live --max-input-tokens 2000000
uv run beyond-answer-confidence report synthetic_worlds         # tables and figures
```

- **Offline is the default.** Requests are answered only from
  `<data_dir>/cache/<experiment>.jsonl`; missing ones are reported, not sent.
- **`--live` costs money.** It sends the uncached requests to the paid API and
  requires `TYPESAFE_API_KEY` to be set and a positive `--max-input-tokens`
  budget. Before each call its own conservative token estimate plus those of
  calls in flight are reserved against the budget, so spending can exceed the
  budget only by estimation error, at most one call's worth per concurrent
  slot (`--concurrency`, 1 to 64, default 8); see [SECURITY.md](SECURITY.md)
  for what the cap does not count. A run stopped by the budget exits with
  status 1. Calls stop after a fatal status (HTTP 401/402/403, e.g. an invalid
  key or no credits). Rerunning resumes from the cache.
- A cache file has a single writer: do not run two live runs of experiments
  that share a cache at the same time.
- The API key is read by the vendor SDK from the `TYPESAFE_API_KEY`
  environment variable; this package never reads or logs it.
- Options come from `configs/<experiment>.toml` (or `--config FILE`) and
  `--set key=value`. The shipped `configs/*.toml` files equal the built-in
  defaults and are optional: they document every option, and a run without
  them uses the same values. Paths default to `./data` and `./outputs`; override them
  with `--data-dir`/`--output-dir` or `BEYOND_ANSWER_CONFIDENCE_DATA_DIR` /
  `BEYOND_ANSWER_CONFIDENCE_OUTPUT_DIR`.
- Offline analyses (comparators, error detection) run with
  `beyond-answer-confidence analyse <name>`.
- `beyond-answer-confidence verify [experiment ...]` rebuilds every request and reports
  how many are in the cache (exit status 1 if any are missing).
- `beyond-answer-confidence export` packages per-item results (ids, labels and
  probabilities only) with a SHA-256 manifest; it refuses any row that could
  carry dataset text.

## Data and licences

Datasets are downloaded at run time from their original sources, at pinned
revisions, and remain under their own licences. Some are non-commercial
(for example ChaosNLI and ANLI, CC BY-NC 4.0); some state no licence.
`beyond-answer-confidence export` writes a table of sources, revisions and licences.

Request caches contain the full prompts, i.e. dataset text: keep them out of
version control and do not redistribute them. `beyond-answer-confidence
check-mirrors` compares the dataset mirrors used with independent copies
(needs network access).

## Security

See [SECURITY.md](SECURITY.md) for how keys are handled and how to report a
vulnerability.

## Disclaimer

This project is not affiliated with or endorsed by TypeSafe.

## Licence

Apache-2.0; see [LICENSE](LICENSE).
