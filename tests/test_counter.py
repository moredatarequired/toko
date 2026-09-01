"""Tests for token counting."""

import pytest

import toko.counter as counter
from tests.cache_keys import cache_key
from toko.cache import cache_count, get_cached_count
from toko.counter import count_tokens
from toko.result import Caveat, CaveatKind, TokenCount

# 13 tokens on cl100k_base and 7 on o200k_base, so a count says which encoding produced
# it. An ASCII phrase would tokenize the same on both and hide an encoding swap.
MIXED_SCRIPT_TEXT = "Café 日本語 🎉 résumé"


def test_count_tokens_simple_text():
    """Test counting tokens in simple text."""
    result = count_tokens("hello world", model="gpt-5")
    assert result.count > 0
    assert result.count == 2


def test_count_tokens_empty_string():
    """Test counting tokens in empty string."""
    result = count_tokens("", model="gpt-5")
    assert result.count == 0


def test_count_tokens_different_models():
    """Test that different models can count the same text."""
    text = "The quick brown fox jumps over the lazy dog"
    gpt5_count = count_tokens(text, model="gpt-5").count
    gpt5_mini_count = count_tokens(text, model="gpt-5-mini").count

    # Both should return positive counts
    assert gpt5_count > 0
    assert gpt5_mini_count > 0
    # gpt-5 and gpt-5-mini share the same tokenizer so counts should match
    assert gpt5_count == gpt5_mini_count


def test_count_tokens_unicode():
    """Test counting tokens with unicode characters."""
    result = count_tokens("Hello 世界", model="gpt-5")
    assert result.count > 0


def test_count_tokens_unknown_model():
    """Test that unknown model raises error."""
    with pytest.raises(ValueError, match="Could not detect provider"):
        count_tokens("hello", model="unknown-model-xyz")


def test_count_tokens_gpt5_variants():
    """Test that gpt-5.x variants use the gpt-5 tokenizer."""
    text = "The quick brown fox jumps over the lazy dog"
    gpt5_count = count_tokens(text, model="gpt-5").count
    gpt51_count = count_tokens(text, model="gpt-5.1").count
    gpt52_count = count_tokens(text, model="gpt-5.2").count

    # All variants should return the same count (same tokenizer)
    assert gpt5_count > 0
    assert gpt51_count == gpt5_count
    assert gpt52_count == gpt5_count


@pytest.mark.parametrize("model", ["gpt-6", "gpt-5.6", "gpt-5.4-mini", "o5"])
def test_unknown_openai_models_estimate_with_o200k_base(model, capsys):
    text = "The quick brown fox jumps over the lazy dog"
    expected = count_tokens(text, model="gpt-5", use_cache=False).count
    capsys.readouterr()

    assert count_tokens(text, model=model, use_cache=False).count == expected

    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        captured.err.strip()
        == f"Warning: unknown OpenAI model '{model}'; estimating with o200k_base"
    )


@pytest.mark.parametrize("model", ["gpt-5", "gpt-4o", "gpt-5-mini", "gpt-5.2"])
def test_exactly_resolved_openai_models_are_not_warned_about(model, capsys):
    count_tokens("hello world", model=model, use_cache=False)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("model", "encoding", "expected"),
    [
        ("gpt-4-1", "cl100k_base", 13),
        ("gpt-4-zzzz", "cl100k_base", 13),
        ("gpt-3.5-turbo-zzzz", "cl100k_base", 13),
        ("gpt-35-turbo-zzzz", "cl100k_base", 13),
        ("gpt-4o-zzzz", "o200k_base", 7),
        ("gpt-5-zzzz", "o200k_base", 7),
        ("o1-zzzz", "o200k_base", 7),
        ("o4-mini-zzzz", "o200k_base", 7),
    ],
)
def test_a_prefix_matched_openai_name_is_estimated_on_its_family_encoding(
    model, encoding, expected, capsys
):
    """Tiktoken answers for anything starting with a family name it knows.

    The encoding is the family's real one, so it is the best count available -- but
    the name itself is unverified, and 'gpt-4-1' is a typo for gpt-4.1 that used to
    come back as an exact cl100k_base count of a name OpenAI never shipped.
    """
    counted = count_tokens(MIXED_SCRIPT_TEXT, model=model, use_cache=False)

    assert counted.count == expected
    assert counted.approximate is True
    assert counted.caveats == (
        Caveat(
            kind=CaveatKind.OPENAI_ENCODING_GUESS,
            model=model,
            message=f"unknown OpenAI model '{model}'; estimating with {encoding}",
            encoding=encoding,
        ),
    )
    assert capsys.readouterr().err.strip() == f"Warning: {counted.caveats[0].message}"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4", 13),
        ("gpt-3.5-turbo", 13),
        ("gpt-4o", 7),
        ("gpt-4.1", 7),
        ("gpt-4.1-mini", 7),
        ("GPT-5", 7),
    ],
)
def test_an_exactly_known_openai_name_stays_exact_and_silent(model, expected, capsys):
    """Demoting the prefix table must not demote a name toko or tiktoken lists."""
    counted = count_tokens(MIXED_SCRIPT_TEXT, model=model, use_cache=False)

    assert (counted.count, counted.approximate, counted.caveats) == (
        expected,
        False,
        (),
    )
    assert capsys.readouterr().err == ""


def test_a_prefix_matched_openai_count_is_not_cached(capsys):
    first = count_tokens(MIXED_SCRIPT_TEXT, model="gpt-4-1")
    counter._WARNED_ONCE.clear()  # noqa: SLF001
    second = count_tokens(MIXED_SCRIPT_TEXT, model="gpt-4-1")

    assert (first.approximate, second.approximate) == (True, True)
    assert get_cached_count(MIXED_SCRIPT_TEXT, "gpt-4-1") is None
    # A cache hit returns before any provider runs, so the second warning is what
    # proves the count was made again rather than replayed as exact.
    assert capsys.readouterr().err.count("Warning:") == 2


def test_a_count_cached_before_a_name_lost_its_exactness_is_not_served(capsys):
    # What an older toko stored for gpt-4-1: an exact count under the bare name,
    # reached through tiktoken's 'gpt-4-' prefix. 4242 is a count no tokenizer
    # produces for this text, so a hit could only have come from the cache.
    cache_count(MIXED_SCRIPT_TEXT, "gpt-4-1", 4242)

    counted = count_tokens(MIXED_SCRIPT_TEXT, model="gpt-4-1")

    assert counted.count == 13
    assert counted.approximate is True
    assert "unknown OpenAI model 'gpt-4-1'" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("spelling", "canonical"), [("GPT-5", "gpt-5"), ("Gpt-5.2", "gpt-5.2")]
)
def test_openai_model_names_resolve_case_insensitively(spelling, canonical, capsys):
    text = "The quick brown fox jumps over the lazy dog"
    expected = count_tokens(text, model=canonical, use_cache=False).count
    capsys.readouterr()

    assert count_tokens(text, model=spelling, use_cache=False).count == expected
    assert capsys.readouterr().err == ""


def test_unknown_openai_model_warns_once(capsys):
    count_tokens("hello", model="gpt-6", use_cache=False)
    count_tokens("goodbye", model="gpt-6", use_cache=False)
    assert capsys.readouterr().err.count("Warning:") == 1


def test_openai_estimate_warns_again_on_a_repeat_run(capsys):
    text = "hello world"

    first = count_tokens(text, model="gpt-6").count
    assert "unknown OpenAI model 'gpt-6'" in capsys.readouterr().err

    # A later invocation is a fresh process, which remembers no warnings.
    counter._WARNED_ONCE.clear()  # noqa: SLF001

    second = count_tokens(text, model="gpt-6").count

    assert second == first
    assert "unknown OpenAI model 'gpt-6'" in capsys.readouterr().err
    assert get_cached_count(text, "gpt-6") is None


def test_warn_once_dedupes_per_kind_not_only_per_model(capsys):
    counter._warn_once("alpha", "some-model", "first notice")  # noqa: SLF001
    counter._warn_once("alpha", "some-model", "first notice")  # noqa: SLF001
    counter._warn_once("beta", "some-model", "second notice")  # noqa: SLF001

    err = capsys.readouterr().err
    assert err.count("Warning:") == 2
    assert "first notice" in err
    assert "second notice" in err


def test_exact_openai_count_is_cached(capsys):
    text = "hello world"
    counted = count_tokens(text, model="gpt-5")
    assert capsys.readouterr().err == ""
    assert get_cached_count(text, cache_key("gpt-5")) == counted.count


def test_count_tokens_reports_the_model_and_provider_it_resolved():
    counted = count_tokens("hello world", model="gpt-5", use_cache=False)

    assert counted == TokenCount(count=2, model="gpt-5", provider="openai")
    assert counted.approximate is False
    assert counted.caveats == ()
    assert counted.cost is None


def test_unknown_openai_model_carries_the_caveat_it_printed(capsys):
    counted = count_tokens("hello world", model="gpt-6", use_cache=False)

    assert counted.approximate is True
    assert counted.caveats == (
        Caveat(
            kind=CaveatKind.OPENAI_ENCODING_GUESS,
            model="gpt-6",
            message="unknown OpenAI model 'gpt-6'; estimating with o200k_base",
            encoding="o200k_base",
        ),
    )
    # The message is the sentence stderr already carried, so the two cannot drift.
    assert capsys.readouterr().err.strip() == f"Warning: {counted.caveats[0].message}"


def test_a_cache_hit_returns_a_fully_populated_count():
    text = "a text no tokenizer has seen in this test"
    # A count no tokenizer would produce, so only the cache can be its source.
    cache_count(text, cache_key("gpt-5"), 4242)

    assert count_tokens(text, model="gpt-5") == TokenCount(
        count=4242, model="gpt-5", provider="openai"
    )
