"""Tests for model metadata utilities."""

from toko import models


def test_list_models_includes_core_providers():
    listed = models.list_models()
    assert "openai" in listed
    assert "anthropic" in listed
    assert "gpt-5" in listed["openai"]


def test_optional_groups_use_module_detection(monkeypatch):
    def fake_has_module(name: str) -> bool:
        return name == "mistral_common"

    monkeypatch.setattr(models, "_has_module", fake_has_module)

    groups = models.list_optional_model_groups()
    extra_status = {g["extra"]: g["installed"] for g in groups}
    assert extra_status["mistral"] is True
    assert extra_status["transformers"] is False
