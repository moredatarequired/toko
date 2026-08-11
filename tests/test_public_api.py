"""The package's public surface, exercised the way a library caller would."""

import pytest

import toko
from toko import TokenCount, count_tokens


def test_count_tokens_is_reachable_from_the_package_root():
    counted = toko.count_tokens("hello world", model="gpt-5")

    assert isinstance(counted, TokenCount)
    assert counted == TokenCount(count=2, model="gpt-5", provider="openai")


def test_the_package_exports_only_the_names_it_advertises():
    assert set(toko.__all__) == {"TokenCount", "__version__", "count_tokens"}
    assert count_tokens is toko.count_tokens


def test_dir_lists_the_lazy_exports_without_hiding_the_module_dunders():
    listed = dir(toko)

    assert set(toko.__all__) <= set(listed)
    # A __dir__ that returned __all__ alone would truncate the module for
    # inspect.getmembers and anything else that walks dir().
    assert "__name__" in listed
    assert listed == sorted(listed)


def test_an_unexported_name_raises_attribute_error():
    # Fetched dynamically so the type checker sees the same unresolved name a caller
    # would be warned about rather than failing this file.
    with pytest.raises(AttributeError, match="module 'toko' has no attribute 'nope'"):
        getattr(toko, "nope")  # noqa: B009
