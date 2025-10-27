"""Shared pytest fixtures and warning filters."""

import warnings

import pytest

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
