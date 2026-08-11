"""Tests for model metadata utilities."""

import pytest

from toko import models


@pytest.mark.parametrize(
    ("name", "provider"),
    [
        ("gpt-6", "openai"),
        ("gpt-5.6", "openai"),
        ("gpt-5.4-mini", "openai"),
        ("o5", "openai"),
        ("gpt-5", "openai"),
        ("gpt-oss-20b", "huggingface"),
        ("openai/gpt-oss-120b", "huggingface"),
        ("claude-opus-4-5", "anthropic"),
        ("unknown-model-xyz", None),
    ],
)
def test_detect_provider(name, provider):
    assert models.detect_provider(name) == provider


def test_unknown_openai_model_carries_no_verified_encoding():
    assert models.get_model("gpt-6") == models.ModelInfo(
        name="gpt-6", provider="openai", encoding=None
    )


def test_dotted_openai_model_keeps_its_own_name_and_encoding():
    assert models.get_model("gpt-5.2") == models.ModelInfo(
        name="gpt-5.2", provider="openai", encoding="o200k_base"
    )


def test_every_priced_verified_encoding_is_advertised():
    advertised = set(models.POPULAR_OPENAI_MODELS)
    verified = set(models.OPENAI_MODEL_ENCODINGS)
    assert verified - advertised == models._UNPRICED_OPENAI_MODELS  # noqa: SLF001
    assert advertised >= set(models._OPENAI_PREFIX_VARIANTS)  # noqa: SLF001


def test_list_models_includes_core_providers():
    listed = models.list_models()
    assert "openai" in listed
    assert "anthropic" in listed
    assert "gpt-5" in listed["openai"]
    assert "gpt-4.1-mini" in listed["openai"]


def test_optional_groups_use_module_detection(monkeypatch):
    def fake_has_module(name: str) -> bool:
        return name == "mistral_common"

    monkeypatch.setattr(models, "_has_module", fake_has_module)

    groups = models.list_optional_model_groups()
    extra_status = {g["extra"]: g["installed"] for g in groups}
    assert extra_status["mistral"] is True
    assert extra_status["transformers"] is False
