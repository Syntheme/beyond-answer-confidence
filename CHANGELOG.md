# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Blog links to other sites whose `href` starts on a new line in the
  source (the arXiv and code links at the top, three footnote links) now
  open in a new tab too.

## [0.1.2] - 2026-10-02

### Added

- `webflow_api.py snippet` prints the chart script tag, with its integrity
  hash, for the blog template's custom code; Webflow's Custom Code API does
  not accept site tokens.

### Changed

- The SmoothECE coverage simulation runs 400 datasets per condition by
  default, as in the published results (it was 300), so a rerun with the
  shipped configuration matches the paper.
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
- Links to other sites in the blog post open in a new tab; `build.py` adds
  `target="_blank" rel="noopener"` to every external link.
- Blog Part 2 opens with a real call, like Part 1: Jev puts 0.68 on an author
  for the made-up book "The Velbri Tide" but answers "Do you know?" with 0.23
  for "yes". A footnote gives the request ID and the unit
  (`knowledge_boundary`, `fab:author:7`) so the call can be rebuilt.
- Blog Part 2's section on asking about the case (now "Asking about the case
  worked better") explains both follow-up questions with an example: a
  two-fact question with complete or incomplete paragraphs, and the made-up
  Roskal Cup final played "last spring" or "next spring".

## [0.1.1] - 2026-10-01

### Added

- Blog post (Part 1 and Part 2) under `assets/blog/`, opening with a
  verifiable fair-die example.
- Webflow build of the blog post (`assets/blog/webflow.py`) and a Data API
  tool (`assets/blog/webflow_api.py`) that pushes the parts to a CMS
  collection as drafts and installs the chart script; dry run by default,
  never publishes.

## [0.1.0] - 2026-10-01

### Added

- First public release.
- Library and command-line interface to probe whether a decision API's
  probabilities track what it knows.
- Cache-first, budget-capped request client with an offline default and an
  explicit `--live` switch for paid calls.
- Deterministic fake backend for tests and the quick start.
- Calibration, ranking and distribution metrics with resampling-based
  inference.

[Unreleased]: https://github.com/Syntheme/beyond-answer-confidence/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/Syntheme/beyond-answer-confidence/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Syntheme/beyond-answer-confidence/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Syntheme/beyond-answer-confidence/releases/tag/v0.1.0
