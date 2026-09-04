"""Small configuration helpers shared by the three filter pipelines."""

from typing import Any


def as_bool(value: Any) -> bool:
    """Convert common boolean-like configuration values to ``bool``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


# Compatibility name retained by the public launch modules.
_as_bool = as_bool
