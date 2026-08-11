"""Tests for model metadata utilities."""

import re

import pytest

from toko import counter, models
from toko.cache import get_cached_count
from toko.counter import count_tokens


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


def test_retirement_notice_carries_no_presentation_prefix():
    notice = models.retirement_notice(models.get_model("grok-3"))
    assert notice is not None
    assert not notice.startswith("Warning")


def test_google_retirement_notice_does_not_leak_the_api_prefix():
    notice = models.retirement_notice(models.get_model("gemini-2.0-flash-001"))
    assert notice is not None
    assert "models/" not in notice
    assert notice.startswith("gemini-2.0-flash-001 was retired")


def test_retired_models_are_never_listed_as_supported():
    listed = {name for names in models.list_models().values() for name in names}
    retired = [
        info.name
        for registry in (
            models.ANTHROPIC_MODELS,
            models.GOOGLE_MODELS,
            models.XAI_MODELS,
        )
        for info in registry.values()
        if info.retired is not None
    ]
    assert retired
    assert not listed.intersection(retired)


class TestXaiRegistry:
    """xAI's published retirement list is the source of truth for these."""

    def test_grok_4_inherits_its_retirement_from_grok_4_0709(self):
        # docs.x.ai retires grok-4-0709, not plain grok-4; grok-4 is the alias
        # for the last stable Grok 4, so its metadata must not be restated.
        assert "grok-4" not in models.XAI_MODELS
        for alias in ("grok-4", "grok-4-latest"):
            resolved = models.get_model(alias)
            assert resolved.name == "grok-4-0709"
            assert resolved.retired == "2026-05-15"
            assert resolved.redirects_to == "grok-4.3"

    def test_the_registry_matches_the_published_retirement_list(self):
        retired_on_may_15 = {
            name
            for name, info in models.XAI_MODELS.items()
            if info.retired == "2026-05-15"
        }
        assert retired_on_may_15 == {
            "grok-4-1-fast-reasoning",
            "grok-4-1-fast-non-reasoning",
            "grok-4-fast-reasoning",
            "grok-4-fast-non-reasoning",
            "grok-4-0709",
            "grok-code-fast-1",
            "grok-3",
            "grok-imagine-image-pro",
        }

    def test_the_current_grok_4_20_family_is_offered(self):
        listed = models.list_models()["xai"]
        for name in (
            "grok-4.20-0309-reasoning",
            "grok-4.20-0309-non-reasoning",
            "grok-4.20-multi-agent-0309",
        ):
            assert name in listed
            assert models.get_model(name).retired is None


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

    def test_a_latest_alias_the_prefix_scan_would_capture_still_passes_through(self):
        """The pass-through only bites on names an alias prefix would have claimed.

        The four aliases above miss the alias map and the prefix scan alike, so
        they resolve verbatim with or without the pass-through. gemini-2.0-flash
        *is* an alias prefix, so gemini-2.0-flash-latest is the case that pins
        the branch: without it the name lands on gemini-2.0-flash-001.
        """
        resolved = models.get_model("gemini-2.0-flash-latest")
        assert resolved.name == "models/gemini-2.0-flash-latest"
        assert resolved.name != "models/gemini-2.0-flash-001"

    def test_a_latest_alias_of_a_retired_family_still_warns(self):
        """Passing through must not lose the retirement caveat.

        Google shut gemini-2.0-flash-001 down on 2026-06-01, but genai-prices
        still prefix-matches the name, so a silent pass-through would price a
        model that no longer exists.
        """
        resolved = models.get_model("gemini-2.0-flash-latest")
        assert resolved.retired == "2026-06-01"
        notice = models.retirement_notice(resolved)
        assert notice is not None
        assert "retired" in notice

    def test_a_latest_alias_of_a_current_family_is_not_warned_about(self):
        assert models.get_model("gemini-flash-latest").retired is None
        assert models.retirement_notice(models.get_model("gemini-flash-latest")) is None

    def test_a_latest_alias_from_another_provider_is_not_claimed(self):
        """The Google resolver runs before xAI's, so it must not swallow -latest."""
        assert models.get_model("grok-4-latest").name == "grok-4-0709"

    def test_a_latest_alias_is_matched_case_insensitively(self):
        assert models.get_model("Gemini-Flash-Latest").name == (
            "models/gemini-flash-latest"
        )

    def test_no_latest_alias_is_pinned_to_a_version(self):
        pinned = [a for a in models._GOOGLE_ALIAS_MAP if a.endswith("-latest")]  # noqa: SLF001
        assert pinned == []

    def test_dated_previews_still_resolve_to_their_stable_release(self):
        assert models.get_model("gemini-2.5-pro-preview-06-05").name == (
            "models/gemini-2.5-pro"
        )


def test_warning_about_a_retired_model_goes_to_stderr(capsys):
    counter._warn_if_retired(models.get_model("claude-3-opus-20240229"))  # noqa: SLF001
    captured = capsys.readouterr()
    assert "retired" in captured.err
    assert captured.out == ""


def test_no_warning_is_emitted_for_a_current_model(capsys):
    counter._warn_if_retired(models.get_model("claude-opus-5"))  # noqa: SLF001
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

    def test_models_on_opposite_sides_never_share_a_cache_key(self, monkeypatch):
        # count_tokens caches under both the name the user typed and the
        # resolved name, so a shorthand that crossed the boundary would serve a
        # 4.6 count for a 4.7 request. Anthropic has no local tokenizer, so the
        # network call is stubbed with a per-generation count -- resolution,
        # cache keying and read-back are all the real code.
        counts = {
            models.CLAUDE_TOKENIZER_LEGACY: 100,
            models.CLAUDE_TOKENIZER_OPUS_4_7: 130,
        }
        monkeypatch.setitem(
            counter._PROVIDER_HANDLERS,  # noqa: SLF001
            "anthropic",
            lambda _text, model_info: counter.CountResult(counts[model_info.tokenizer]),
        )

        text = "the boundary must hold"
        assert count_tokens(text, "claude-opus-4-6") == 100
        assert count_tokens(text, "claude-opus-4-6-latest") == 100
        assert count_tokens(text, "claude-opus-4-7") == 130
        assert count_tokens(text, "claude-opus-5") == 130

        # The 4.7 counts must not have landed on a 4.6 key on the way through.
        assert count_tokens(text, "claude-opus-4-6") == 100
        assert get_cached_count(text, "claude-opus-4-6-latest") == 100

    def test_a_shorthand_spanning_the_boundary_is_left_unresolvable(self, monkeypatch):
        # No dated ID in the shipped registry sits on the new tokenizer, so the
        # alias map's boundary guard is only reachable with an injected
        # registry -- but the day such an ID ships, the shorthand must not
        # silently pick a side.
        def spec(date: str, tokenizer: str):
            name = f"claude-fictional-9-{date}"
            return name, models.ModelInfo(
                name=name, provider="anthropic", tokenizer=tokenizer
            )

        spanning = dict(
            [
                spec("20260101", models.CLAUDE_TOKENIZER_LEGACY),
                spec("20260601", models.CLAUDE_TOKENIZER_OPUS_4_7),
            ]
        )
        monkeypatch.setattr(models, "ANTHROPIC_MODELS", spanning)
        assert "claude-fictional-9" not in models._build_anthropic_alias_map()  # noqa: SLF001

        within = dict(
            [
                spec("20260101", models.CLAUDE_TOKENIZER_LEGACY),
                spec("20260601", models.CLAUDE_TOKENIZER_LEGACY),
            ]
        )
        monkeypatch.setattr(models, "ANTHROPIC_MODELS", within)
        assert models._build_anthropic_alias_map() == {  # noqa: SLF001
            "claude-fictional-9": "claude-fictional-9-20260601"
        }

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
