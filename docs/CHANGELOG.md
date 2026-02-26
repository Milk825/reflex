# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Reflex (formerly autonomy-system)
- Real-time tool interception via TypeScript plugin hooks
- Python-based confidence scoring with 6-signal formula
- Pattern detection (looping, over-eager, confidence collapse)
- Three operation modes: Shadow, Suggest, Active
- Circuit breaker for failing tools
- Meta-cognition monitoring
- Self-evaluation and reflection
- 76 comprehensive tests (58 Python + 18 TypeScript)
- Unix Domain Socket bridge for sub-10ms latency
- SQLite + JSONL telemetry storage
- Fail-open safety design
- MIT License

### Security
- Fixed ASK_CONFIRMATION bypass (session-scoped tracking)
- Fixed failOpen config being ignored in error paths
- Fixed unsafe socket cleanup (validates socket before removal)
- Fixed mode-control inconsistency (runtime mode switching)

## [0.1.0] - 2026-02-26

### Added
- Initial public release
- GitHub repository: https://github.com/Milk825/reflex
- Complete rebranding from "autonomy-system" to "Reflex"
- Professional README with shields/badges
- Logo (SVG + PNG)
- Social preview image (1200x630)

[Unreleased]: https://github.com/Milk825/reflex/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Milk825/reflex/releases/tag/v0.1.0