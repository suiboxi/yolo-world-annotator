# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses semantic versioning.

## [Unreleased]

## [0.0.1] - 2026-08-30

Initial open-source release.

### Added

- Automatic CUDA/CPU selection with explicit `cpu` and `cuda:N` overrides.
- Standard `src/` package layout, CLI entry point, packaging metadata, and CPU CI.
- Cross-platform user-data, log, and weight paths.
- GitHub contribution, security, issue, pull request, and dependency-update files.

### Changed

- Model loading no longer starts merely by opening an advanced-window project.
- Windows launch and build scripts no longer contain a personal Python path.
- Windows builds no longer bundle large model weights by default.
