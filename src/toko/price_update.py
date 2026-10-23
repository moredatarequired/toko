"""Automatic price update handling."""

import contextlib
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import httpx
from genai_prices import Usage, data_snapshot, types
from genai_prices.update_prices import DEFAULT_UPDATE_URL

from toko.cache import get_cache_dir

PRICE_DATA_URL = DEFAULT_UPDATE_URL
FETCH_TIMEOUT = httpx.Timeout(timeout=10, connect=5)
_PROBE_USAGE = Usage(input_tokens=1_000_000, output_tokens=1_000_000)


def get_price_cache_path() -> Path:
    """Get the path to the price update timestamp cache.

    Returns:
        Path to timestamp file
    """
    return get_cache_dir() / "price_update_timestamp"


def get_price_data_path() -> Path:
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
        last_update = float(timestamp_file.read_text(encoding="utf-8").strip())
        age = time.time() - last_update
    except (ValueError, OSError):
        # If we can't read the timestamp, assume we should update
        return True
    else:
        return age > max_age_seconds


def _has_usable_price(providers: list[types.Provider]) -> bool:
    # The payload comes from a URL that evolves independently of the pinned library, so
    # it can parse cleanly and still price everything at zero once its price keys drift
    # past what this genai-prices understands. Silent $0.00 is worse than a fallback.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for provider in providers:
            for model in provider.models:
                try:
                    priced = model.calc_price(_PROBE_USAGE, provider)
                except Exception:  # noqa: S112
                    continue
                # Per-token input and output, specifically: keys priced per request
                # can survive a rename of the token keys and mask the drift.
                if priced.input_price > 0 and priced.output_price > 0:
                    return True
    return False


def _build_snapshot(payload: bytes) -> data_snapshot.DataSnapshot:
    # genai-prices only parses price payloads inside UpdatePrices.fetch(), which always
    # downloads them and hands back a snapshot rather than the bytes we need to cache.
    # fetch() reaches for this same private parser, so there is no public alternative;
    # pyproject caps genai-prices below 0.2 to keep the symbol from moving under us.
    from genai_prices.types import _providers_from_raw  # noqa: PLC0415

    providers = _providers_from_raw(json.loads(payload))
    if not providers:
        raise ValueError("Pricing data contains no providers")
    if not _has_usable_price(providers):
        raise ValueError("Pricing data has no usable per-token prices")
    return data_snapshot.DataSnapshot(providers, from_auto_update=True)


def _write_atomic(path: Path, payload: bytes) -> None:
    # Concurrent toko processes share this cache, so a half-written file must never be
    # observable: write a sibling temp file and swap it in with a single rename.
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as temp_file:
            temp_file.write(payload)
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def apply_cached_prices() -> bool:
    """Apply the previously fetched price data, if a usable copy is cached.

    Returns:
        True if cached prices were applied, False otherwise
    """
    data_path = get_price_data_path()
    try:
        payload = data_path.read_bytes()
    except FileNotFoundError:
        return False

    try:
        snapshot = _build_snapshot(payload)
    except Exception as e:
        print(
            f"Warning: discarding unusable cached price data at {data_path}"
            f" ({type(e).__name__}: {e}). Falling back to bundled prices;"
            " run 'toko update-prices' to refetch.",
            file=sys.stderr,
        )
        # Without this the warning repeats on every run forever: a refetch only
        # overwrites the file when the new payload validates, so a cache poisoned by
        # the remote itself would never be replaced.
        with contextlib.suppress(OSError):
            data_path.unlink(missing_ok=True)
        return False

    data_snapshot.set_custom_snapshot(snapshot)
    return True


def clear_price_cache() -> None:
    for path in (get_price_data_path(), get_price_cache_path()):
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


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

    data_snapshot.set_custom_snapshot(snapshot)
    _write_atomic(get_price_data_path(), response.content)
    _write_atomic(get_price_cache_path(), str(time.time()).encode())
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
