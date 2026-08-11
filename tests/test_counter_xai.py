"""Tests for xAI token counting strategy."""

import httpx
import pytest

import toko.counter as counter


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

    tokens = counter.count_tokens("hello", "grok-4.5", use_cache=False)
    assert tokens == 7
    # An exact API count must not be labelled approximate.
    assert capsys.readouterr().err == ""


def test_xai_falls_back_to_transformers(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    def fake_post(*_args, **_kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(counter.httpx, "post", fake_post)
    _install_stub_tokenizer(monkeypatch)

    tokens = counter.count_tokens("hi", "grok-4.5", use_cache=False)
    assert tokens == len("hi")


def test_xai_api_failure_warns_that_count_is_approximate(monkeypatch, capsys):
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    def fake_post(*_args, **_kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(counter.httpx, "post", fake_post)
    _install_stub_tokenizer(monkeypatch)

    tokens = counter.count_tokens("hi", "grok-4.5", use_cache=False)

    stderr = capsys.readouterr().err
    assert tokens == len("hi")
    assert "grok-4.5" in stderr
    assert "approximate" in stderr
    assert "boom" in stderr


def test_xai_without_api_key_warns_that_count_is_approximate(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    _install_stub_tokenizer(monkeypatch)

    tokens = counter.count_tokens("hi", "grok-4.5", use_cache=False)

    stderr = capsys.readouterr().err
    assert tokens == len("hi")
    assert "XAI_API_KEY is not set" in stderr
    assert "approximate" in stderr


def test_xai_approximation_warning_is_not_repeated(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    _install_stub_tokenizer(monkeypatch)

    for text in ("one", "two", "three"):
        counter.count_tokens(text, "grok-4.5", use_cache=False)

    assert capsys.readouterr().err.count("approximate") == 1


class TestRetirementWarningsThroughCountTokens:
    """count_tokens must caveat a retired model's count however it was obtained.

    grok-3 is served by grok-4.3, so its number is deliberately mislabelled --
    the caveat is the only thing that says so.
    """

    @pytest.fixture(autouse=True)
    def _offline_xai(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        _install_stub_tokenizer(monkeypatch)

    def test_a_cached_count_still_warns_that_the_model_is_retired(self, capsys):
        assert counter.count_tokens("hi", "grok-3") == len("hi")
        assert "retired" in capsys.readouterr().err

        # The dedup set is process-global, so clearing it is what a second
        # `toko` invocation looks like -- except the cache is now warm.
        counter._RETIRED_WARNED.clear()  # noqa: SLF001
        assert counter.count_tokens("hi", "grok-3") == len("hi")
        assert "retired" in capsys.readouterr().err

    def test_the_retirement_warning_is_not_repeated_within_a_run(self, capsys):
        for text in ("one", "two", "three"):
            counter.count_tokens(text, "grok-3")

        assert capsys.readouterr().err.count("was retired") == 1

    def test_a_current_model_is_never_warned_about(self, capsys):
        counter.count_tokens("hi", "grok-4.5")
        assert "retired" not in capsys.readouterr().err

    def test_an_alias_of_a_retired_model_inherits_the_warning(self, capsys):
        counter.count_tokens("hi", "grok-4")
        stderr = capsys.readouterr().err
        assert "grok-4-0709 was retired" in stderr
        assert "grok-4.3" in stderr


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
