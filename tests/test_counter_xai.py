"""Tests for xAI token counting strategy."""

import httpx
import pytest

import toko.counter as counter
from toko.cache import get_cached_count


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


@pytest.fixture(autouse=True)
def _reset_approximation_warnings():
    counter._APPROXIMATE_WARNED.clear()  # noqa: SLF001
    yield
    counter._APPROXIMATE_WARNED.clear()  # noqa: SLF001


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

    tokens = counter.count_tokens("hello", "grok-3", use_cache=False)
    assert tokens == 7
    # An exact API count must not be labelled approximate.
    assert capsys.readouterr().err == ""


def test_xai_falls_back_to_transformers(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    def fake_post(*_args, **_kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(counter.httpx, "post", fake_post)
    _install_stub_tokenizer(monkeypatch)

    tokens = counter.count_tokens("hi", "grok-3", use_cache=False)
    assert tokens == len("hi")


def test_xai_api_failure_warns_that_count_is_approximate(monkeypatch, capsys):
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    def fake_post(*_args, **_kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(counter.httpx, "post", fake_post)
    _install_stub_tokenizer(monkeypatch)

    tokens = counter.count_tokens("hi", "grok-3", use_cache=False)

    stderr = capsys.readouterr().err
    assert tokens == len("hi")
    assert "grok-3" in stderr
    assert "approximate" in stderr
    assert "boom" in stderr


def test_xai_without_api_key_warns_that_count_is_approximate(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    _install_stub_tokenizer(monkeypatch)

    tokens = counter.count_tokens("hi", "grok-3", use_cache=False)

    stderr = capsys.readouterr().err
    assert tokens == len("hi")
    assert "XAI_API_KEY is not set" in stderr
    assert "approximate" in stderr


def test_xai_approximation_warning_is_not_repeated(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    _install_stub_tokenizer(monkeypatch)

    for text in ("one", "two", "three"):
        counter.count_tokens(text, "grok-3", use_cache=False)

    assert capsys.readouterr().err.count("approximate") == 1


def test_xai_approximate_count_warns_again_on_a_repeat_run(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    _install_stub_tokenizer(monkeypatch)

    first = counter.count_tokens("hi", "grok-3")
    assert "approximate" in capsys.readouterr().err

    # A later invocation is a fresh process, which remembers no warnings.
    counter._APPROXIMATE_WARNED.clear()  # noqa: SLF001

    second = counter.count_tokens("hi", "grok-3")

    assert second == first
    assert "approximate" in capsys.readouterr().err
    assert get_cached_count("hi", "grok-3") is None


def test_xai_api_count_is_cached(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    def fake_post(*_args, **_kwargs):
        return DummyResponse({"token_count": 7})

    monkeypatch.setattr(counter.httpx, "post", fake_post)

    # Ensure fallback is not invoked
    monkeypatch.setattr(counter, "_count_xai_via_transformers", lambda _text: 0)

    assert counter.count_tokens("hello", "grok-3") == 7
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
        counter.count_tokens("hello", "grok-3", use_cache=False)
