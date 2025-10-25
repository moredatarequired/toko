"""Automatic price update handling."""

import tempfile
import time
from pathlib import Path

from genai_prices import UpdatePrices


def get_price_cache_path() -> Path:
    """Get the path to the price update timestamp cache.

    Returns:
        Path to timestamp file
    """
    # Store in the same directory as genai_prices data
    # This is a simple approach - we just track when we last checked
    cache_dir = Path(tempfile.gettempdir()) / "toko"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / "price_update_timestamp"


def should_update_prices(max_age_seconds: int = 86400) -> bool:
    """Check if prices should be updated based on staleness.

    Args:
        max_age_seconds: Maximum age in seconds (default 86400 = 1 day)

    Returns:
        True if prices should be updated, False otherwise
    """
    timestamp_file = get_price_cache_path()

    if not timestamp_file.exists():
        return True

    try:
        last_update = float(timestamp_file.read_text().strip())
        age = time.time() - last_update
    except (ValueError, OSError):
        # If we can't read the timestamp, assume we should update
        return True
    else:
        return age > max_age_seconds


def update_prices_if_stale(max_age_seconds: int = 86400) -> bool:
    """Update prices if they are stale.

    Args:
        max_age_seconds: Maximum age in seconds (default 86400 = 1 day)

    Returns:
        True if prices were updated, False if they were already fresh
    """
    if not should_update_prices(max_age_seconds):
        return False

    updater = UpdatePrices()
    result = updater.fetch()

    if result:
        # Update timestamp
        timestamp_file = get_price_cache_path()
        timestamp_file.write_text(str(time.time()))
        return True

    return False
