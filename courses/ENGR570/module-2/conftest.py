"""
pytest configuration for Module 2.

Importing astronomical_time applies the offline IERS policy (no downloads of
Earth-orientation or leap-second data) before any test runs.
"""

import astronomical_time  # noqa: F401  (import for its side effect)
