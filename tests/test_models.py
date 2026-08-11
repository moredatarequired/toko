"""Tests for model metadata utilities."""

import re

import pytest

from toko import models
from toko.cache import cache_count, clear_cache, get_cached_count


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


def test_current_models_are_listed_and_retired_ones_are_hidden():
    listed = models.list_models()
    assert "claude-opus-5" in listed["anthropic"]
    assert "claude-sonnet-5" in listed["anthropic"]
    assert "claude-fable-5" in listed["anthropic"]
    assert "models/gemini-3.1-pro-preview" in listed["google"]
    assert "grok-4.5" in listed["xai"]

    assert "claude-3-opus-20240229" not in listed["anthropic"]
    assert "models/gemini-2.0-flash-001" not in listed["google"]
    assert "grok-3" not in listed["xai"]

    with_retired = models.list_models(include_retired=True)
    assert "claude-3-opus-20240229" in with_retired["anthropic"]
    assert "models/gemini-2.0-flash-001" in with_retired["google"]
    assert "grok-3" in with_retired["xai"]


@pytest.mark.parametrize(
    "name",
    ["claude-3-opus-20240229", "claude-opus-4-1-20250805", "gemini-2.0-flash-001"],
)
def test_retired_models_still_resolve_and_report_their_retirement(name):
    notice = models.retirement_notice(models.get_model(name))
    assert notice is not None
    assert "retired" in notice


def test_current_models_report_no_retirement():
    for name in ["claude-opus-5", "gemini-2.5-pro", "grok-4.5"]:
        assert models.retirement_notice(models.get_model(name)) is None


def test_silently_redirected_model_says_whose_count_it_is():
    notice = models.retirement_notice(models.get_model("grok-4"))
    assert notice is not None
    # xAI answers for grok-4 with grok-4.3, so the count must not be presented
    # as grok-4's.
    assert "grok-4.3" in notice


class TestGoogleLatestAliases:
    """Google repoints its "-latest" aliases on two weeks' notice.

    Pinning a target here means reporting some other model's count the moment
    Google moves it, so the alias is sent to the API verbatim.
    """

    @pytest.mark.parametrize(
        "alias",
        [
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
            "gemini-pro-latest",
            "gemini-2.5-flash-native-audio-latest",
        ],
    )
    def test_a_latest_alias_is_passed_through_verbatim(self, alias):
        resolved = models.get_model(alias)
        assert resolved.provider == "google"
        assert resolved.name == f"models/{alias}"

    def test_no_latest_alias_is_pinned_to_a_version(self):
        pinned = [a for a in models._GOOGLE_ALIAS_MAP if a.endswith("-latest")]  # noqa: SLF001
        assert pinned == []

    def test_dated_previews_still_resolve_to_their_stable_release(self):
        assert models.get_model("gemini-2.5-pro-preview-06-05").name == (
            "models/gemini-2.5-pro"
        )


def test_warning_about_a_retired_model_goes_to_stderr(capsys):
    models.warn_if_retired(models.get_model("claude-3-opus-20240229"))
    captured = capsys.readouterr()
    assert "retired" in captured.err
    assert captured.out == ""


def test_no_warning_is_emitted_for_a_current_model(capsys):
    models.warn_if_retired(models.get_model("claude-opus-5"))
    assert capsys.readouterr().err == ""


class TestAnthropicTokenizerBoundary:
    """Guard the Claude Opus 4.7 tokenizer boundary.

    Opus 4.7 changed tokenizer, so the same text counts ~30% higher on
    4.7-generation models. Nothing may map a name across that boundary.
    """

    def test_the_registry_actually_spans_the_boundary(self):
        tokenizers = {m.tokenizer for m in models.ANTHROPIC_MODELS.values()}
        assert tokenizers == {
            models.CLAUDE_TOKENIZER_LEGACY,
            models.CLAUDE_TOKENIZER_OPUS_4_7,
        }

    def test_models_on_opposite_sides_never_resolve_to_each_other(self):
        older = models.get_model("claude-opus-4-6")
        newer = models.get_model("claude-opus-4-7")
        assert older.tokenizer != newer.tokenizer
        assert older.name == "claude-opus-4-6"
        assert newer.name == "claude-opus-4-7"

    def test_models_on_opposite_sides_never_share_a_cache_key(self, cache_dir):
        clear_cache()
        assert cache_dir.exists()
        text = "the boundary must hold"
        # count_tokens caches under both the name the user asked for and the
        # resolved name, so a 4.6 count must never be served for a 4.7 request.
        for name in ("claude-opus-4-6", "claude-opus-4-6-latest"):
            cache_count(text, models.get_model(name).name, 100)

        assert get_cached_count(text, "claude-opus-4-6") == 100
        assert get_cached_count(text, "claude-opus-4-7") is None
        assert get_cached_count(text, "claude-opus-5") is None

    def test_no_shorthand_resolves_across_the_boundary(self):
        """Every name that resolves must land on its own tokenizer generation."""
        for name, info in models.ANTHROPIC_MODELS.items():
            undated = re.sub(r"-\d{8}$", "", name)
            for candidate in (name, undated, f"{undated}-latest", f"{name}-latest"):
                resolved = models.get_model(candidate)
                assert resolved.tokenizer == info.tokenizer, (
                    f"{candidate} resolved to {resolved.name}, which is on the "
                    f"other side of the Opus 4.7 tokenizer change from {name}"
                )

    @pytest.mark.parametrize(
        "name", ["claude-opus-4-6-latest", "claude-opus-4-7-latest"]
    )
    def test_latest_suffix_stays_model_exact(self, name):
        assert models.get_model(name).name == name.removesuffix("-latest")
