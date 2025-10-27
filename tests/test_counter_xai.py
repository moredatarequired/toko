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


def test_xai_prefers_api(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    def fake_post(*_args, **_kwargs):
        return DummyResponse({"token_count": 7})

    monkeypatch.setattr(counter.httpx, "post", fake_post)

    # Ensure fallback is not invoked
    monkeypatch.setattr(counter, "_count_xai_via_transformers", lambda _text: 0)

    tokens = counter.count_tokens("hello", "grok-3", use_cache=False)
    assert tokens == 7


def test_xai_falls_back_to_transformers(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    def fake_post(*_args, **_kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(counter.httpx, "post", fake_post)

    class StubTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(text.encode())

    monkeypatch.setattr(counter, "HAS_TRANSFORMERS", True)
    monkeypatch.setitem(
        counter._TOKENIZER_CACHE,  # noqa: SLF001
        "transformers:xai:grok-1",
        StubTokenizer(),
    )

    tokens = counter.count_tokens("hi", "grok-3", use_cache=False)
    assert tokens == len("hi")


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
