# Runtime State

This directory is the default local home for mutable runtime state.

Examples:

- trained model registry SQLite database
- model artifacts
- request audit log
- drift baseline snapshot
- model promotion registry

The directory is intentionally outside source packages so local state does not live under
importable application code.
