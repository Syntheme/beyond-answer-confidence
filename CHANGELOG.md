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
- The blog post and the READMEs link to the paper on arXiv
  (<https://arxiv.org/abs/2610.01006>) instead of the PDF in the repository;
  the README has a BibTeX entry for citing it.
- Blog Part 2, from the rerun against the paper: the stated-odds error is
  "about 3 points" (2.97), and when few questions come from beyond the
  knowledge boundary the confidence threshold does "a little better" than
  the "not known" option, rather than winning outright (the rerun's
  interval includes zero).
- Blog prose, chart titles and captions, and diagram labels edited for a
  plainer style: fewer colons used as connectors, no rhetorical questions or
  filler adverbs, straight quotes, words instead of "=" in chart captions.
  No numbers or findings changed. `blog.js` changed, so the publishing
  guide's script line points at the next tag, v0.1.2.
- Blog section headings are a single numbered line ("1. We changed what Jev
  could know"): the label above each heading ("01 · The controlled test") is
  gone, and every section, including the conclusion and methods, is numbered
  to match the contents list.
- The three boxes under "Methods, limitations and provenance" start open.

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
