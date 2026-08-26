"""Shared pytest fixtures and warning filters."""

import warnings

import pytest
from genai_prices.data_snapshot import set_custom_snapshot

import toko.counter as counter
from toko.cache import clear_cache, set_cache_dir

# Suppress noisy DeprecationWarnings emitted by optional tokenizer backends.
warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPyPacked has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPyObject has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
)


@pytest.fixture
def isolated_git_env(monkeypatch, tmp_path):
    """Keep a test off the developer's real git config and home directory."""
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture(autouse=True)
def _cache_root(tmp_path):
    cache_root = tmp_path / "cache"
    set_cache_dir(cache_root)
    try:
        yield cache_root
    finally:
        clear_cache()
        set_cache_dir(None)


@pytest.fixture(autouse=True)
def _bundled_prices():
    """Keep fetched prices from leaking between tests."""
    try:
        yield
    finally:
        set_custom_snapshot(None)


@pytest.fixture
def cache_dir(_cache_root):
    return _cache_root


@pytest.fixture(autouse=True)
def _reset_warned_models():
    """Start every test from an empty warned-once registry.

    It is process-global, so a "warns once" assertion is otherwise order-dependent.
    """
    counter._WARNED_ONCE.clear()  # noqa: SLF001
    yield
    counter._WARNED_ONCE.clear()  # noqa: SLF001
