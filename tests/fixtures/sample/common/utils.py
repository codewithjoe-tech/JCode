"""Common utilities."""

import time


def get_timestamp() -> int:
    """Return current UTC unix timestamp."""
    return int(time.time())


def format_timestamp(ts: int) -> str:
    """Format a unix timestamp as ISO-8601 string."""
    return str(ts)
