"""
Configuration module for S2Doc.

This module provides centralized configuration management for performance tuning,
logging, and other runtime parameters. Values can be overridden via environment variables.

Environment Variables:
    - S2DOC_UUID_BATCH_SIZE: Number of UUIDs to pre-generate in cache (default: 5)
    - S2DOC_LAZY_GEOMETRY_ENABLED: Enable lazy loading of geometry objects (default: true)
    - S2DOC_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
"""

import os
from typing import Final

# Performance tuning
UUID_BATCH_SIZE: Final[int] = int(os.getenv("S2DOC_UUID_BATCH_SIZE", "5"))
"""Number of UUIDs to pre-generate and cache for batch generation."""

LAZY_GEOMETRY_ENABLED: Final[bool] = os.getenv(
    "S2DOC_LAZY_GEOMETRY_ENABLED", "true"
).lower() in ("true", "1", "yes")
"""Enable lazy loading of geometry objects to defer computation."""

# Logging configuration
LOG_LEVEL: Final[str] = os.getenv("S2DOC_LOG_LEVEL", "INFO")
"""Default logging level for S2Doc modules."""

def get_config_summary() -> dict[str, bool | int | str | float]:
    """
    Return a dictionary of current configuration values.

    Returns:
        dict: All configuration parameters and their current values.
    """
    return {
        "UUID_BATCH_SIZE": UUID_BATCH_SIZE,
        "LAZY_GEOMETRY_ENABLED": LAZY_GEOMETRY_ENABLED,
        "LOG_LEVEL": LOG_LEVEL,
    }
