# Security policy

## Supported versions

Only the latest release receives security fixes.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Report it privately through GitHub: open the repository's **Security** tab and
choose **Report a vulnerability** (GitHub private vulnerability reporting).
Include what you found, how to reproduce it, and the impact you expect. You
should get an acknowledgement within a few working days; we will keep you
updated while a fix is prepared and credit you in the release notes unless you
prefer otherwise.

## How API keys are handled

`beyond-answer-confidence` can call a paid decision API. The design goal is that the key
never passes through this package and that nothing is spent by accident.

- **Environment only.** The key is read from the `TYPESAFE_API_KEY`
  environment variable by the vendor SDK itself. This package never reads,
  stores, prints or logs the value, and it never accepts a key on the command
  line or in a config file.
- **Quiet SDK logging.** The SDK logger is pinned at `WARNING`, because request
  headers carry the key. Request bodies are never logged at `INFO`.
- **Offline by default.** Every command answers from the local response cache
  (or the deterministic fake backend) unless you pass `--live`. Without
  `--live`, no network call to the API is made.
- **Key presence check.** `run --live` refuses to start unless
  `TYPESAFE_API_KEY` is set and non-empty; only its presence is checked.
- **Budget cap.** Live runs enforce a cap on new input tokens. Before each
  call, its own conservative token estimate (request characters / 1.5) plus
  the estimates of calls already in flight are reserved against the cap, and
  the call is refused if the cap would be exceeded. The cap is therefore not a
  hard guarantee: spending can exceed it by at most one call's estimation
  error (actual minus estimated tokens) per concurrent slot (`--concurrency`,
  1 to 64). A run stopped by the budget exits with status 1. Authentication or
  billing errors (HTTP 401, 402, 403) stop the run immediately instead of
  retrying.
- **What the cap counts.** Only responses that arrive and are recorded in the
  cache count against the cap (a paid response that fails to parse is still
  recorded and counted). The SDK retries after timeouts and transient errors;
  if the provider processed a request whose response never arrived, a retry
  can bill it twice, and the first charge is not counted. Keep the cap below
  what you are prepared to spend, and check the provider's usage dashboard.
- **Nothing sensitive in the repository.** `.gitignore` denies `.env` files and
  key material, and every commit is checked by `detect-secrets`, `gitleaks`,
  `detect-private-key` and a generic leak audit (`tools/leak-audit.sh`:
  credentials, e-mail addresses, home-directory paths, `.env` files, notebook
  outputs, large files). CI repeats gitleaks and the leak audit over the full
  history and runs `detect-secrets` over every tracked file.

If you think a key has leaked, revoke it with the provider first; removing it
from git history does not make it safe again.
