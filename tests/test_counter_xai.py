"""Tests for xAI token counting strategy."""

import httpx
import pytest

import toko.counter as counter
from toko.cache import get_cached_count
from toko.result import CaveatKind, Retirement


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        """Do nothing."""

    def json(self):
        """Return payload."""
        return self._payload


class StubTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode())


def _install_stub_tokenizer(monkeypatch):
    monkeypatch.setattr(counter, "HAS_TRANSFORMERS", True)
    monkeypatch.setitem(
        counter._TOKENIZER_CACHE,  # noqa: SLF001
        "transformers:xai:grok-1",
        StubTokenizer(),
    )


def test_xai_prefers_api(monkeypatch, capsys):
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    def fake_post(*_args, **_kwargs):
        return DummyResponse({"token_count": 7})

    monkeypatch.setattr(counter.httpx, "post", fake_post)

    # Ensure fallback is not invoked
    monkeypatch.setattr(counter, "_count_xai_via_transformers", lambda _text: 0)

    counted = counter.count_tokens("hello", "grok-4.5", use_cache=False)
    assert counted.count == 7
    assert counted.approximate is False
    # An exact API count must not be labelled approximate.
    assert capsys.readouterr().err == ""


def test_xai_falls_back_to_transformers(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    def fake_post(*_args, **_kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(counter.httpx, "post", fake_post)
    _install_stub_tokenizer(monkeypatch)

    counted = counter.count_tokens("hi", "grok-4.5", use_cache=False)
    assert counted.count == len("hi")


def test_xai_api_failure_warns_that_count_is_approximate(monkeypatch, capsys):
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    def fake_post(*_args, **_kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(counter.httpx, "post", fake_post)
    _install_stub_tokenizer(monkeypatch)

    counted = counter.count_tokens("hi", "grok-4.5", use_cache=False)

    stderr = capsys.readouterr().err
    assert counted.count == len("hi")
    assert counted.approximate is True
    assert counted.caveats[0].kind is CaveatKind.XAI_GROK1_STANDIN
    assert "grok-4.5" in stderr
    assert "approximate" in stderr
    # The transport's own message is never quoted, only what kind of failure it was.
    assert "HTTPError contacting https://api.x.ai/v1/tokenize" in stderr


def test_xai_without_api_key_warns_that_count_is_approximate(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    _install_stub_tokenizer(monkeypatch)

    counted = counter.count_tokens("hi", "grok-4.5", use_cache=False)

    stderr = capsys.readouterr().err
    assert counted.count == len("hi")
    assert counted.approximate is True
    assert counted.caveats[0].message == stderr.removeprefix("Warning: ").strip()
    assert "XAI_API_KEY is not set" in stderr
    assert "approximate" in stderr


def test_xai_approximation_warning_is_not_repeated(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    _install_stub_tokenizer(monkeypatch)

    for text in ("one", "two", "three"):
        counter.count_tokens(text, "grok-4.5", use_cache=False)

    assert capsys.readouterr().err.count("approximate") == 1


def test_xai_approximate_count_warns_again_on_a_repeat_run(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    _install_stub_tokenizer(monkeypatch)

    first = counter.count_tokens("hi", "grok-3").count
    assert "approximate" in capsys.readouterr().err

    # A later invocation is a fresh process, which remembers no warnings.
    counter._WARNED_ONCE.clear()  # noqa: SLF001

    second = counter.count_tokens("hi", "grok-3").count

    assert second == first
    assert "approximate" in capsys.readouterr().err
    assert get_cached_count("hi", "grok-3") is None


class TestRetirementWarningsThroughCountTokens:
    """count_tokens must caveat a retired model's count however it was obtained.

    grok-3 is served by grok-4.3, so its number is deliberately mislabelled --
    the caveat is the only thing that says so.
    """

    @pytest.fixture(autouse=True)
    def _offline_xai(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        _install_stub_tokenizer(monkeypatch)

    def test_a_cached_count_still_warns_that_the_model_is_retired(
        self, monkeypatch, capsys
    ):
        # Only exact counts are cached, so the API has to answer here; the
        # transformers fallback the class installs is deliberately bypassed.
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr(
            counter.httpx, "post", lambda *_a, **_k: DummyResponse({"token_count": 2})
        )

        assert counter.count_tokens("hi", "grok-3").count == 2
        assert "retired" in capsys.readouterr().err
        assert get_cached_count("hi", "grok-3") == 2

        # The dedup set is process-global, so clearing it is what a second
        # `toko` invocation looks like -- except the cache is now warm.
        counter._WARNED_ONCE.clear()  # noqa: SLF001
        assert counter.count_tokens("hi", "grok-3").count == 2
        assert "retired" in capsys.readouterr().err

    def test_the_retirement_warning_is_not_repeated_within_a_run(self, capsys):
        for text in ("one", "two", "three"):
            counter.count_tokens(text, "grok-3")

        assert capsys.readouterr().err.count("was retired") == 1

    def test_a_current_model_is_never_warned_about(self, capsys):
        counter.count_tokens("hi", "grok-4.5")
        assert "retired" not in capsys.readouterr().err

    def test_a_retired_count_carries_a_structured_retirement(self, capsys):
        counted = counter.count_tokens("hi", "grok-3")

        assert counted.retirement == Retirement(
            model="grok-3", date="2026-05-15", redirects_to="grok-4.3"
        )
        # The stderr warning is unchanged; the field is what a program reads.
        assert "was retired" in capsys.readouterr().err

    def test_a_cache_hit_still_carries_the_retirement(self, monkeypatch, capsys):
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr(
            counter.httpx, "post", lambda *_a, **_k: DummyResponse({"token_count": 2})
        )

        assert counter.count_tokens("hi", "grok-3").count == 2
        capsys.readouterr()

        cached = counter.count_tokens("hi", "grok-3")

        assert cached.count == 2
        assert cached.retirement == Retirement(
            model="grok-3", date="2026-05-15", redirects_to="grok-4.3"
        )

    def test_a_current_model_carries_no_retirement(self):
        assert counter.count_tokens("hi", "grok-4.5").retirement is None

    def test_an_alias_of_a_retired_model_inherits_the_warning(self, capsys):
        counter.count_tokens("hi", "grok-4")
        stderr = capsys.readouterr().err
        assert "grok-4-0709 was retired" in stderr
        assert "grok-4.3" in stderr


def test_xai_api_count_is_cached(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    def fake_post(*_args, **_kwargs):
        return DummyResponse({"token_count": 7})

    monkeypatch.setattr(counter.httpx, "post", fake_post)

    # Ensure fallback is not invoked
    monkeypatch.setattr(counter, "_count_xai_via_transformers", lambda _text: 0)

    assert counter.count_tokens("hello", "grok-3").count == 7
    assert get_cached_count("hello", "grok-3") == 7


def test_xai_failure_without_options(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(counter, "HAS_TRANSFORMERS", False)
    monkeypatch.delitem(
        counter._TOKENIZER_CACHE,  # noqa: SLF001
        "transformers:xai:grok-1",
        raising=False,
    )

    with pytest.raises(ValueError, match="XAI_API_KEY"):
        counter.count_tokens("hello", "grok-4.5", use_cache=False)
