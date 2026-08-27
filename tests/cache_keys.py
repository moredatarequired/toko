"""Cache keys as count_tokens composes them, for tests that read the cache directly."""

from toko import counter
from toko.models import get_model


def cache_key(model: str) -> str:
    key = counter._cache_key(model, get_model(model))  # noqa: SLF001
    assert key is not None, f"counts for {model} are never cached"
    return key
