"""Automatic price update handling."""

import json
import time
from typing import TYPE_CHECKING

import httpx
from genai_prices import UpdatePrices, data_snapshot

from toko.cache import get_cache_dir

if TYPE_CHECKING:
    from pathlib import Path

PRICE_DATA_URL = UpdatePrices().url
FETCH_TIMEOUT = httpx.Timeout(timeout=10, connect=5)


def get_price_cache_path() -> Path:
    """Get the path to the price update timestamp cache.

    Returns:
        Path to timestamp file
    """
    return get_cache_dir() / "price_update_timestamp"


def get_price_data_path() -> Path:
    """Get the path to the cached copy of the fetched price data.

    Returns:
        Path to the cached price payload
    """
    return get_cache_dir() / "prices.json"


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


def _build_snapshot(payload: bytes) -> data_snapshot.DataSnapshot:
    # genai-prices only parses price payloads inside UpdatePrices.fetch(), which always
    # downloads them, so reuse its parser directly to rebuild a snapshot from disk.
    from genai_prices.types import _providers_from_raw  # noqa: PLC0415

    providers = _providers_from_raw(json.loads(payload))
    return data_snapshot.DataSnapshot(providers, from_auto_update=True)


def apply_cached_prices() -> bool:
    """Apply the previously fetched price data, if a usable copy is cached.

    Returns:
        True if cached prices were applied, False otherwise
    """
    try:
        snapshot = _build_snapshot(get_price_data_path().read_bytes())
    except Exception:
        return False

    data_snapshot.set_custom_snapshot(snapshot)
    return True


def refresh_prices() -> int:
    """Fetch the latest prices, apply them, and cache them for later runs.

    The freshness timestamp is only written once a snapshot has been applied, so a
    failed update is retried rather than silently suppressed for a day.

    Returns:
        Number of providers in the fetched data

    Raises:
        httpx.HTTPError: If the price data could not be downloaded.
        ValueError: If the downloaded data is unusable.
    """
    response = httpx.get(PRICE_DATA_URL, timeout=FETCH_TIMEOUT)
    response.raise_for_status()

    snapshot = _build_snapshot(response.content)
    if not snapshot.providers:
        raise ValueError("Fetched pricing data contains no providers")

    data_snapshot.set_custom_snapshot(snapshot)
    get_price_data_path().write_bytes(response.content)
    get_price_cache_path().write_text(str(time.time()))
    return len(snapshot.providers)


def update_prices_if_stale(max_age_seconds: int = 86400) -> bool:
    """Fetch and apply the latest prices if the cached copy has gone stale.

    Callers are expected to have already run apply_cached_prices(); this only decides
    whether a fresh download is due.

    Args:
        max_age_seconds: Maximum age in seconds (default 86400 = 1 day)

    Returns:
        True if prices were fetched, False if the cached data was still fresh

    Raises:
        httpx.HTTPError: If the price data could not be downloaded.
        ValueError: If the downloaded data is unusable.
    """
    if not should_update_prices(max_age_seconds):
        return False

    refresh_prices()
    return True
