# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Webflow build of the blog post (`assets/blog/webflow.py`) and a Data API
  tool (`assets/blog/webflow_api.py`) that pushes the parts to a CMS
  collection as drafts and installs the chart script; dry run by default,
  never publishes.
- `webflow_api.py snippet` prints the chart script tag, with its integrity
  hash, for the blog template's custom code; Webflow's Custom Code API does
  not accept site tokens.

### Changed

- Blog post titles, summaries and Webflow slugs follow synthpop.ai's series
  style ("… (Post 1)", `-post-1`); `webflow_api.py push` links the parts
  under `/resources/` by default. The publishing guide carries the chart
  script line, checked against the build by a test.

## [0.1.0] - Unreleased

### Added

- First public release.
- Library and command-line interface to probe whether a decision API's
  probabilities track what it knows.
- Cache-first, budget-capped request client with an offline default and an
  explicit `--live` switch for paid calls.
- Deterministic fake backend for tests and the quick start.
- Calibration, ranking and distribution metrics with resampling-based
  inference.
