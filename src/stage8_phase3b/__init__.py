"""Phase 3B confirmatory replay utilities.

The package is intentionally guard-heavy: it prepares and validates replay
manifests, then delegates any approved replay to the original Stage 8 v4 code
path.  Importing this package must never start training or inference.
"""

