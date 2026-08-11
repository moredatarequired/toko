"""Tests for price_update module."""

import time
from pathlib import Path

import httpx
import pytest
import respx
from genai_prices.data_snapshot import get_snapshot, set_custom_snapshot

from toko.price_update import (
    PRICE_DATA_URL,
    get_price_cache_path,
    get_price_data_path,
    refresh_prices,
    should_update_prices,
    update_prices_if_stale,
)


def test_get_price_cache_path():
    """Test getting the cache path."""
    path = get_price_cache_path()
    assert isinstance(path, Path)
    assert path.name == "price_update_timestamp"


def test_should_update_prices_no_file():
    """Test that we should update if no timestamp file exists."""
    # Clean up any existing timestamp
    timestamp_file = get_price_cache_path()
    if timestamp_file.exists():
        timestamp_file.unlink()

    assert should_update_prices()


def test_should_update_prices_stale():
    """Test that we should update if timestamp is stale."""
    timestamp_file = get_price_cache_path()
    timestamp_file.parent.mkdir(exist_ok=True, parents=True)

    # Write a timestamp from 2 days ago
    old_time = time.time() - (2 * 86400)
    timestamp_file.write_text(str(old_time))

    assert should_update_prices(max_age_seconds=86400)

    # Clean up
    timestamp_file.unlink()


def test_should_update_prices_fresh():
    """Test that we should not update if timestamp is fresh."""
    timestamp_file = get_price_cache_path()
    timestamp_file.parent.mkdir(exist_ok=True, parents=True)

    # Write current timestamp
    timestamp_file.write_text(str(time.time()))

    assert not should_update_prices(max_age_seconds=86400)

    # Clean up
    timestamp_file.unlink()


def test_update_prices_if_stale():
    """Test updating prices if stale."""
    timestamp_file = get_price_cache_path()
    if timestamp_file.exists():
        timestamp_file.unlink()

    # Should update since no timestamp exists
    result = update_prices_if_stale()
    assert result is True

    # Should not update immediately after
    result = update_prices_if_stale()
    assert result is False

    # Clean up
    if timestamp_file.exists():
        timestamp_file.unlink()


def test_refresh_prices_applies_fetched_snapshot():
    """Fetched prices must actually replace the bundled ones."""
    assert get_snapshot().from_auto_update is False

    provider_count = refresh_prices()

    assert provider_count > 0
    assert get_snapshot().from_auto_update is True
    assert len(get_snapshot().providers) == provider_count


def test_cached_prices_survive_into_a_fresh_process():
    """A later run should reuse the cached payload instead of fetching again."""
    refresh_prices()
    assert get_price_data_path().read_bytes()

    # Simulate a fresh process, which starts out on the bundled price data.
    set_custom_snapshot(None)
    assert get_snapshot().from_auto_update is False

    assert update_prices_if_stale() is False
    assert get_snapshot().from_auto_update is True


@respx.mock
def test_failed_download_leaves_prices_stale():
    """A failed fetch must not mark prices fresh for the next day."""
    respx.get(PRICE_DATA_URL).mock(return_value=httpx.Response(503))

    with pytest.raises(httpx.HTTPError):
        update_prices_if_stale()

    assert not get_price_cache_path().exists()
    assert should_update_prices()
    assert get_snapshot().from_auto_update is False


@respx.mock
def test_empty_payload_leaves_prices_stale():
    """Data we cannot use must not count as a successful update."""
    respx.get(PRICE_DATA_URL).mock(return_value=httpx.Response(200, json=[]))

    with pytest.raises(ValueError, match="no providers"):
        refresh_prices()

    assert not get_price_cache_path().exists()
    assert not get_price_data_path().exists()
    assert should_update_prices()
    assert get_snapshot().from_auto_update is False
