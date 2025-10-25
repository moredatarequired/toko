"""Tests for price_update module."""

import time
from pathlib import Path

from toko.price_update import (
    get_price_cache_path,
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
