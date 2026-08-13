"""Tests for cache module."""

from concurrent.futures import ThreadPoolExecutor

from toko.cache import cache_count, clear_cache, get_cache_db_path, get_cached_count


def test_cache_and_retrieve(cache_dir):
    """Test caching and retrieving token counts."""
    # Clear cache first
    clear_cache()
    assert cache_dir.exists()

    text = "hello world"
    model = "gpt-4.1"
    count = 42

    # Should not be cached initially
    assert get_cached_count(text, model) is None

    # Cache it
    cache_count(text, model, count)

    # Should now be cached
    assert get_cached_count(text, model) == count


def test_cache_multiple_models(cache_dir):
    """Test caching multiple models for same text."""
    clear_cache()
    assert cache_dir.exists()

    text = "hello world"
    model1 = "gpt-4.1"
    model2 = "claude-4-5-haiku-20241022"
    count1 = 42
    count2 = 84

    # Cache for both models
    cache_count(text, model1, count1)
    cache_count(text, model2, count2)

    # Both should be cached
    assert get_cached_count(text, model1) == count1
    assert get_cached_count(text, model2) == count2


def test_cache_different_texts(cache_dir):
    """Test caching different texts."""
    clear_cache()
    assert cache_dir.exists()

    text1 = "hello world"
    text2 = "goodbye world"
    model = "gpt-4.1"
    count1 = 42
    count2 = 84

    # Cache both texts
    cache_count(text1, model, count1)
    cache_count(text2, model, count2)

    # Both should be cached independently
    assert get_cached_count(text1, model) == count1
    assert get_cached_count(text2, model) == count2


def test_clear_cache(cache_dir):
    """Test clearing the cache."""
    clear_cache()
    assert cache_dir.exists()

    text = "hello world"
    model = "gpt-4.1"
    count = 42

    # Cache it
    cache_count(text, model, count)
    assert get_cached_count(text, model) == count

    # Clear cache
    clear_cache()

    # Should no longer be cached
    assert get_cached_count(text, model) is None


def test_get_cache_db_path(cache_dir):
    """Test getting the cache database path."""
    path = get_cache_db_path()
    assert path.name == "token_cache.db"
    assert path.parent == cache_dir


def test_counts_racing_for_one_message_all_survive():
    """Several models counted concurrently against one text share a single row."""
    text = "hello world"
    models = [f"model-{index}" for index in range(16)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda model: cache_count(text, model, len(model)), models))

    assert {model: get_cached_count(text, model) for model in models} == {
        model: len(model) for model in models
    }
