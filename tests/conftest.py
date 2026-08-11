"""Shared pytest fixtures and warning filters."""

import warnings

import pytest

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
warnings.filterwarnings(
    "ignore",
    message="Calling `MistralTokenizer.from_model(..., strict=False)` is deprecated",
    category=FutureWarning,
)


@pytest.fixture(autouse=True)
def _cache_root(tmp_path):
    cache_root = tmp_path / "cache"
    set_cache_dir(cache_root)
    try:
        yield cache_root
    finally:
        clear_cache()
        set_cache_dir(None)


@pytest.fixture
def cache_dir(_cache_root):
    return _cache_root


@pytest.fixture(autouse=True)
def _reset_warned_models():
    """Start every test from an empty set of already-warned-about models.

    Both registries are process-global, so a "warns once" assertion is otherwise
    order-dependent.
    """
    registries = (counter._APPROXIMATE_WARNED, counter._RETIRED_WARNED)  # noqa: SLF001
    for registry in registries:
        registry.clear()
    yield
    for registry in registries:
        registry.clear()
