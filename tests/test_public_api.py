"""The package's public surface, exercised the way a library caller would."""

import toko
from toko import TokenCount, count_tokens


def test_count_tokens_is_reachable_from_the_package_root():
    counted = toko.count_tokens("hello world", model="gpt-5")

    assert isinstance(counted, TokenCount)
    assert counted == TokenCount(count=2, model="gpt-5", provider="openai")


def test_the_package_exports_only_the_names_it_advertises():
    assert set(toko.__all__) == {"TokenCount", "__version__", "count_tokens"}
    assert count_tokens is toko.count_tokens
