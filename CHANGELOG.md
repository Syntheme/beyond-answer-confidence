# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
