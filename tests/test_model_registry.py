"""Tests for the TOML model registry and its user overlay."""

import importlib
import tomllib
from importlib import resources
from types import SimpleNamespace

import pytest

from toko import models
from toko.counter import count_tokens


@pytest.fixture
def user_registry(tmp_path, monkeypatch):
    """Write a real ~/.config/toko/models.toml and reload the registry."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config_dir = tmp_path / "toko"
    config_dir.mkdir()
    overlay = config_dir / "models.toml"

    def write(text: str | bytes):
        if isinstance(text, bytes):
            overlay.write_bytes(text)
        else:
            overlay.write_text(text)
        return importlib.reload(models)

    yield write

    monkeypatch.undo()
    importlib.reload(models)


def _registry(*documents: str) -> models.Registry:
    return models.build_registry(
        [(f"document {i}", tomllib.loads(text)) for i, text in enumerate(documents)]
    )


def _registry_listing(*, include_retired: bool = False) -> dict[str, list[str]]:
    """Keep only the --list-models entries that models.toml is responsible for.

    tiktoken and the optional transformers extra contribute the rest, and both
    move independently of this repo.
    """
    from_registry = {info.name for info in models.MODELS.values()}
    return {
        provider: listed
        for provider, names in models.list_models(
            include_retired=include_retired
        ).items()
        if (listed := [name for name in names if name in from_registry])
    }


GOLDEN_HINT = (
    "src/toko/data/models.toml no longer produces the --list-models entries this "
    "test pins. If you added, retired or unlisted a model on purpose, update "
    "GOLDEN_LISTING / GOLDEN_LISTING_WITH_RETIRED below to match."
)

GOLDEN_LISTING = {
    "anthropic": [
        "claude-fable-5",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-5-20251101",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-6",
        "claude-sonnet-5",
    ],
    "google": [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-flash-image",
        "models/gemini-2.5-flash-lite",
        "models/gemini-2.5-pro",
        "models/gemini-3.1-flash-lite",
        "models/gemini-3.1-pro-preview",
        "models/gemini-3.5-flash",
        "models/gemini-3.5-flash-lite",
        "models/gemini-3.6-flash",
    ],
    "openai": [
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-instruct",
        "gpt-4-32k",
        "gpt-4-turbo",
        "gpt-4-vision-preview",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4.5-preview",
        "gpt-4o-audio-preview",
        "gpt-4o-mini",
        "gpt-4o-mini-audio-preview",
        "gpt-4o-mini-realtime-preview",
        "gpt-4o-mini-transcribe",
        "gpt-4o-mini-tts",
        "gpt-4o-realtime-preview",
        "gpt-4o-search-preview",
        "gpt-4o-transcribe",
        "gpt-5-image",
        "gpt-5-image-mini",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-pro",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.2-pro",
        "o1-mini",
        "o1-pro",
        "o3-deep-research",
        "o3-mini",
        "o3-pro",
        "o4-mini-deep-research",
    ],
    "xai": [
        "grok-4.20-0309-non-reasoning",
        "grok-4.20-0309-reasoning",
        "grok-4.20-multi-agent-0309",
        "grok-4.3",
        "grok-4.5",
        "grok-build-0.1",
    ],
}

GOLDEN_LISTING_WITH_RETIRED = {
    "anthropic": [
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229",
        "claude-fable-5",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-opus-4-5-20251101",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-4-20250514",
        "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-6",
        "claude-sonnet-5",
    ],
    "google": [
        "models/gemini-2.0-flash-001",
        "models/gemini-2.0-flash-lite-001",
        "models/gemini-2.0-flash-preview-image-generation",
        "models/gemini-2.5-flash",
        "models/gemini-2.5-flash-image",
        "models/gemini-2.5-flash-lite",
        "models/gemini-2.5-pro",
        "models/gemini-3-pro-preview",
        "models/gemini-3.1-flash-lite",
        "models/gemini-3.1-flash-lite-preview",
        "models/gemini-3.1-pro-preview",
        "models/gemini-3.5-flash",
        "models/gemini-3.5-flash-lite",
        "models/gemini-3.6-flash",
    ],
    "openai": [
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-instruct",
        "gpt-4-32k",
        "gpt-4-turbo",
        "gpt-4-vision-preview",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4.5-preview",
        "gpt-4o-audio-preview",
        "gpt-4o-mini",
        "gpt-4o-mini-audio-preview",
        "gpt-4o-mini-realtime-preview",
        "gpt-4o-mini-transcribe",
        "gpt-4o-mini-tts",
        "gpt-4o-realtime-preview",
        "gpt-4o-search-preview",
        "gpt-4o-transcribe",
        "gpt-5-image",
        "gpt-5-image-mini",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-pro",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.2-pro",
        "o1-mini",
        "o1-pro",
        "o3-deep-research",
        "o3-mini",
        "o3-pro",
        "o4-mini-deep-research",
    ],
    "xai": [
        "grok-2-1212",
        "grok-2-vision-1212",
        "grok-3",
        "grok-3-mini",
        "grok-4-0709",
        "grok-4-1-fast-non-reasoning",
        "grok-4-1-fast-reasoning",
        "grok-4-fast-non-reasoning",
        "grok-4-fast-reasoning",
        "grok-4.20-0309-non-reasoning",
        "grok-4.20-0309-reasoning",
        "grok-4.20-multi-agent-0309",
        "grok-4.3",
        "grok-4.5",
        "grok-build-0.1",
        "grok-code-fast-1",
        "grok-imagine-image-pro",
    ],
}


class TestPackagedRegistry:
    def test_the_registry_ships_inside_the_package(self):
        resource = resources.files("toko.data").joinpath(models.REGISTRY_FILENAME)
        assert resource.is_file(), (
            f"{models.REGISTRY_FILENAME} is not installed alongside toko, so the "
            "build no longer packages it and toko knows no models"
        )

    @pytest.mark.parametrize("provider", ["anthropic", "google", "xai", "openai"])
    def test_every_provider_has_models(self, provider):
        assert models.load_registry().models[provider]

    def test_every_anthropic_model_declares_a_known_tokenizer(self):
        """The Opus 4.7 boundary is only enforceable if the data carries it.

        A missing or misspelled `tokenizer` in models.toml would silently turn
        the boundary guard in _build_anthropic_alias_map into a no-op.
        """
        known = {models.CLAUDE_TOKENIZER_LEGACY, models.CLAUDE_TOKENIZER_OPUS_4_7}
        for name, info in models.ANTHROPIC_MODELS.items():
            assert info.tokenizer in known, (
                f"{name} carries tokenizer {info.tokenizer!r}, so nothing stops "
                "alias resolution from crossing the Opus 4.7 tokenizer change"
            )

    def test_google_models_keep_their_api_prefix(self):
        assert models.GOOGLE_MODELS["gemini-2.5-pro"].name == "models/gemini-2.5-pro"

    def test_retirement_metadata_survives_the_registry(self):
        # grok-4 is a declared alias of grok-4-0709, so this also proves the
        # alias inherits the retirement rather than restating it.
        grok_4 = models.get_model("grok-4")
        assert grok_4.retired == "2026-05-15"
        assert grok_4.redirects_to == "grok-4.3"
        assert grok_4.encoding == "o200k_base"

    def test_the_listing_is_exactly_what_the_registry_declares(self):
        assert _registry_listing() == GOLDEN_LISTING, GOLDEN_HINT

    def test_retired_models_appear_only_when_asked_for(self):
        assert _registry_listing(include_retired=True) == GOLDEN_LISTING_WITH_RETIRED, (
            GOLDEN_HINT
        )

    def test_no_advertised_openai_model_leans_on_tiktokens_prefix_table(self, capsys):
        """A prefix table entry claims a family, not a name.

        tiktoken resolves anything starting with "gpt-5-" through that prefix, so
        an entry with no encoding of its own counted exactly only by accident.
        Toko now estimates those and says so, which for a model --list-models
        advertises is a warning on every run: declare the encoding instead.
        """
        estimated = {
            name: counted.caveat
            for name in sorted(models.list_models()["openai"])
            if (counted := count_tokens("hello world", model=name, use_cache=False))
            and counted.approximate
        }
        capsys.readouterr()
        assert estimated == {}, (
            "these models are advertised but counted approximately; give each an "
            f"encoding in src/toko/data/models.toml: {sorted(estimated)}"
        )

    def test_the_packaged_registry_loads_without_a_word_on_stderr(self, capsys):
        """Every warning reports a user's edit, so an overlay-free run is quiet.

        This is also what keeps a duplicate alias inside models.toml itself from
        being invisible: the warning for one now fires only when a declaration
        loses to an earlier document, which no packaged entry can do.
        """
        models.build_registry([models._load_packaged_document()])  # noqa: SLF001
        assert capsys.readouterr().err == ""

    def test_a_packaged_registry_that_is_not_utf8_fails_loudly(self, monkeypatch):
        """An install bug must surface as a clear error, not a decode traceback."""
        corrupt = SimpleNamespace(read_bytes=lambda: b'[[model]]\nname = "\xff"\n')
        monkeypatch.setattr(
            models.resources,
            "files",
            lambda _: SimpleNamespace(joinpath=lambda _: corrupt),
        )
        with pytest.raises(RuntimeError, match="corrupt"):
            models._load_packaged_document()  # noqa: SLF001


class TestRegistryParsing:
    def test_every_field_survives_a_round_trip(self):
        registry = _registry("""
            [[model]]
            name = "claude-imaginary-9"
            provider = "anthropic"
            tokenizer = "claude-opus-4-7"
            encoding = "o200k_base"
            api_endpoint = "https://example.invalid/count"
            retired = "2030-01-01"
            redirects_to = "claude-opus-5"
            listed = false
        """)
        assert registry.models["anthropic"]["claude-imaginary-9"] == models.ModelInfo(
            name="claude-imaginary-9",
            provider="anthropic",
            encoding="o200k_base",
            api_endpoint="https://example.invalid/count",
            retired="2030-01-01",
            redirects_to="claude-opus-5",
            tokenizer="claude-opus-4-7",
            listed=False,
        )

    def test_a_later_document_overrides_field_by_field(self):
        registry = _registry(
            """
            [[model]]
            name = "claude-opus-4-6"
            provider = "anthropic"
            tokenizer = "claude-legacy"
            """,
            """
            [[model]]
            name = "claude-opus-4-6"
            retired = "2030-01-01"
            """,
        )
        overridden = registry.models["anthropic"]["claude-opus-4-6"]
        assert overridden.retired == "2030-01-01"
        # Merging by field is what keeps the override from dropping tokenizer.
        assert overridden.tokenizer == "claude-legacy"

    def test_an_entry_with_no_provider_is_skipped(self, capsys):
        registry = _registry("""
            [[model]]
            name = "who-knows-1"
        """)
        assert registry.models == {}
        assert "who-knows-1" in capsys.readouterr().err

    def test_malformed_and_unknown_fields_are_skipped_not_fatal(self, capsys):
        registry = _registry("""
            [[model]]
            name = "grok-imaginary-9"
            provider = "xai"
            encoding = 12
            retried = "2030-01-01"
        """)
        assert registry.models["xai"]["grok-imaginary-9"].encoding is None
        stderr = capsys.readouterr().err
        assert "encoding" in stderr
        assert "retried" in stderr

    def test_a_name_declared_twice_in_one_document_is_flagged(self, capsys):
        registry = _registry("""
            [[model]]
            name = "grok-imaginary-9"
            provider = "xai"
            encoding = "o200k_base"

            [[model]]
            name = "grok-imaginary-9"
            provider = "xai"
            retired = "2030-01-01"
        """)
        merged = registry.models["xai"]["grok-imaginary-9"]
        assert (merged.encoding, merged.retired) == ("o200k_base", "2030-01-01")
        assert "grok-imaginary-9" in capsys.readouterr().err

    def test_alias_lists_accumulate_instead_of_replacing(self):
        registry = _registry(
            """
            [[model]]
            name = "grok-imaginary-9"
            provider = "xai"
            aliases = ["grok-a", "grok-b"]
            """,
            """
            [[model]]
            name = "grok-imaginary-9"
            aliases = ["grok-b", "grok-c"]
            """,
        )
        assert registry.aliases["xai"] == {
            "grok-a": "grok-imaginary-9",
            "grok-b": "grok-imaginary-9",
            "grok-c": "grok-imaginary-9",
        }

    def test_an_alias_key_is_registered_lowercased(self):
        """Every lookup lowercases, so a capitalised key would never match."""
        registry = _registry("""
            [[model]]
            name = "grok-imaginary-9"
            provider = "xai"
            aliases = ["Grok-Nickname"]
        """)
        assert registry.aliases["xai"] == {"grok-nickname": "grok-imaginary-9"}

    def test_declaring_one_alias_on_two_models_says_which_won(self, capsys):
        """Registry order decides, so the loser must not lose silently."""
        registry = _registry("""
            [[model]]
            name = "grok-imaginary-8"
            provider = "xai"
            aliases = ["grok-shared"]

            [[model]]
            name = "grok-imaginary-9"
            provider = "xai"
            aliases = ["grok-shared"]
        """)
        assert registry.aliases["xai"]["grok-shared"] == "grok-imaginary-9"

        err = capsys.readouterr().err
        assert "grok-shared" in err
        assert "grok-imaginary-8" in err
        assert "grok-imaginary-9" in err

    def test_a_single_owner_alias_is_not_warned_about(self, capsys):
        _registry("""
            [[model]]
            name = "grok-imaginary-9"
            provider = "xai"
            aliases = ["grok-solo"]
        """)
        assert "grok-solo" not in capsys.readouterr().err

    def test_an_alias_that_repeats_a_model_name_is_ignored_out_loud(self, capsys):
        """A model name is matched before any alias table, across providers."""
        registry = _registry("""
            [[model]]
            name = "grok-imaginary-8"
            provider = "xai"

            [[model]]
            name = "grok-imaginary-9"
            provider = "xai"
            aliases = ["grok-imaginary-8"]
        """)
        assert registry.aliases.get("xai", {}) == {}

        err = capsys.readouterr().err
        assert "grok-imaginary-8" in err
        assert "grok-imaginary-9" in err

    def test_only_aliasable_providers_may_declare_aliases(self, capsys):
        registry = _registry("""
            [[model]]
            name = "claude-opus-5"
            provider = "anthropic"
            tokenizer = "claude-opus-4-7"
            aliases = ["claude-opus-4-6-latest"]
        """)
        assert registry.aliases.get("anthropic") is None
        assert "aliases" in capsys.readouterr().err


class TestGoogleAliasPrefixes:
    """An unrecognized variant must fall back to its closest alias, not any."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("gemini-2.0-flash-lite-preview-11-11", "models/gemini-2.0-flash-lite-001"),
            ("gemini-2.0-flash-exp-11-11", "models/gemini-2.0-flash-001"),
            (
                "gemini-2.0-flash-exp-image-generation-11-11",
                "models/gemini-2.0-flash-preview-image-generation",
            ),
            ("gemini-2.5-flash-lite-preview-11-11", "models/gemini-2.5-flash-lite"),
        ],
    )
    def test_the_longest_matching_alias_wins(self, name, expected):
        assert models.get_model(name).name == expected

    def test_a_prefix_only_matches_on_a_separator(self):
        """gemini-2.0-flash-lite must not claim a name that merely starts with it."""
        resolved = models.get_model("gemini-2.0-flash-litex")
        assert resolved.name == "models/gemini-2.0-flash-001"

    def test_a_glued_character_falls_through_to_the_api(self):
        """With no shorter alias to catch it, the name goes to Google verbatim.

        gemini-exp-1206 is an alias of gemini-2.5-pro, but nothing may claim
        gemini-exp-1206x on its behalf -- a made-up name must reach the API and
        be rejected there rather than be counted as a real, different model.
        """
        assert models.get_model("gemini-exp-1206").name == "models/gemini-2.5-pro"
        assert models.get_model("gemini-exp-1206x").name == "models/gemini-exp-1206x"

    def test_resolution_does_not_depend_on_registry_order(self, user_registry):
        """Appending an alias must not re-route names that already resolved."""
        before = models.get_model("gemini-2.0-flash-lite-preview-11-11").name
        reloaded = user_registry("""
            [[model]]
            name = "gemini-2.5-pro"
            provider = "google"
            aliases = ["gemini-2.0-flash-imaginary"]
        """)
        assert reloaded.get_model("gemini-2.0-flash-lite-preview-11-11").name == before


class TestUserOverlay:
    def test_a_new_model_is_picked_up(self, user_registry):
        reloaded = user_registry("""
            [[model]]
            name = "claude-imaginary-9"
            provider = "anthropic"
            tokenizer = "claude-opus-4-7"
        """)
        assert reloaded.get_model("claude-imaginary-9").provider == "anthropic"
        assert "claude-imaginary-9" in reloaded.list_models()["anthropic"]

    def test_an_override_keeps_the_packaged_fields_it_does_not_name(
        self, user_registry
    ):
        reloaded = user_registry("""
            [[model]]
            name = "claude-opus-4-6"
            retired = "2030-01-01"
        """)
        overridden = reloaded.get_model("claude-opus-4-6")
        assert overridden.retired == "2030-01-01"
        assert overridden.tokenizer == reloaded.CLAUDE_TOKENIZER_LEGACY

    def test_an_override_extends_the_shipped_aliases(self, user_registry):
        """Adding an alias must not unpublish the ones the registry shipped."""
        reloaded = user_registry("""
            [[model]]
            name = "gemini-2.5-pro"
            provider = "google"
            aliases = ["my-pro"]
        """)
        assert reloaded.get_model("my-pro").name == "models/gemini-2.5-pro"
        # Dropping a shipped alias does not raise: the name falls through to the
        # generic Google builder and is sent to the API under an ID it does not
        # serve, so nothing but this assertion would notice.
        assert reloaded.get_model("gemini-exp-1206").name == "models/gemini-2.5-pro"

    def test_a_capitalised_user_alias_is_reachable(self, user_registry):
        """A nickname typed with capitals must resolve, under either casing."""
        reloaded = user_registry("""
            [[model]]
            name = "gemini-2.5-pro"
            provider = "google"
            aliases = ["Gemini-Pro-Nick"]
        """)
        assert reloaded.get_model("Gemini-Pro-Nick").name == "models/gemini-2.5-pro"
        assert reloaded.get_model("gemini-pro-nick").name == "models/gemini-2.5-pro"

    def test_re_pointing_a_shipped_alias_backwards_is_not_silent(
        self, user_registry, capsys
    ):
        """Re-pointing only works forwards, so the failing direction must say so.

        gemini-exp-1206 ships on gemini-2.5-pro. gemini-3.1-pro-preview is
        declared earlier in the packaged registry, so claiming the alias there
        loses on order -- which the user must be told, not left to discover
        through a wrong count.
        """
        reloaded = user_registry("""
            [[model]]
            name = "gemini-3.1-pro-preview"
            provider = "google"
            aliases = ["gemini-exp-1206"]
        """)
        assert reloaded.get_model("gemini-exp-1206").name == "models/gemini-2.5-pro"

        err = capsys.readouterr().err
        assert "gemini-exp-1206" in err
        assert "gemini-3.1-pro-preview" in err
        assert "gemini-2.5-pro" in err

    def test_re_pointing_a_shipped_alias_forwards_takes_effect(
        self, user_registry, capsys
    ):
        """gemini-2.5-flash is declared after gemini-2.5-pro, so it wins.

        Silently: this is the documented, supported way to re-point an alias,
        and warning about it would nag on every invocation -- including runs
        that never mention Google.
        """
        reloaded = user_registry("""
            [[model]]
            name = "gemini-2.5-flash"
            provider = "google"
            aliases = ["gemini-exp-1206"]
        """)
        assert reloaded.get_model("gemini-exp-1206").name == "models/gemini-2.5-flash"
        assert capsys.readouterr().err == ""

    def test_a_new_model_name_is_the_reliable_way_to_re_point_an_alias(
        self, user_registry, capsys
    ):
        """A name the packaged registry lacks is appended last, so it always wins.

        The overlay claims one alias twice and lists the doomed entry first.
        Overriding a packaged model inherits that model's position, so the
        user's own file order decides nothing -- which is why the README sends
        people to a brand-new name.
        """
        reloaded = user_registry("""
            [[model]]
            name = "gemini-3.1-pro-preview"
            provider = "google"
            aliases = ["gemini-exp-1206"]

            [[model]]
            name = "my-own-gemini"
            provider = "google"
            aliases = ["gemini-exp-1206"]
        """)
        assert reloaded.get_model("gemini-exp-1206").name == "models/my-own-gemini"

        # Only the declaration that lost is reported, and the winner it names
        # has to be the model that actually ends up with the alias. A third
        # claimant sits between the two in merged order, so a message built
        # from whoever held the alias mid-walk would name gemini-2.5-pro and
        # send the user to fix a model that is working.
        err = capsys.readouterr().err
        assert "declared on 'gemini-3.1-pro-preview' has no effect" in err
        assert "'my-own-gemini' declares it too" in err
        assert "so 'my-own-gemini' keeps it" in err
        assert "gemini-2.5-pro" not in err

    def test_a_capitalised_model_name_is_reachable_under_either_spelling(
        self, user_registry
    ):
        """A capitalised entry must not answer to its own spelling alone."""
        reloaded = user_registry("""
            [[model]]
            name = "Gemini-My-Model"
            provider = "google"
            retired = "2099-01-01"
        """)
        for spelling in ("Gemini-My-Model", "gemini-my-model"):
            resolved = reloaded.get_model(spelling)
            assert resolved.name == "models/gemini-my-model"
            assert resolved.retired == "2099-01-01"

        listed = reloaded.list_models(include_retired=True)["google"]
        assert "models/gemini-my-model" in listed
        assert "models/Gemini-My-Model" not in listed

    def test_a_user_encoding_wins_over_tiktokens_prefix_table(self, user_registry):
        """The documented per-model `encoding` override, on a name tiktoken claims.

        tiktoken resolves "gpt-4-1" through its "gpt-4-" prefix to cl100k_base, which
        used to answer before the registry was consulted at all -- so the one field a
        user has for teaching toko an encoding did nothing for any name a prefix
        covered. "Café 日本語 🎉 résumé" is 13 tokens on cl100k_base and 7 on
        o200k_base, so the count alone says which encoding was used.
        """
        reloaded = user_registry("""
            [[model]]
            name = "gpt-4-1"
            provider = "openai"
            encoding = "o200k_base"
        """)
        assert reloaded.get_model("gpt-4-1").encoding == "o200k_base"

        counted = count_tokens(
            "Café 日本語 🎉 résumé", model="gpt-4-1", use_cache=False
        )

        assert counted.count == 7
        assert counted.approximate is False
        assert counted.caveat is None

    def test_a_capitalised_openai_name_keeps_its_registry_metadata(self, user_registry):
        """OpenAI has no lowercasing resolver of its own, unlike the others.

        Without the registry lookup itself being case-insensitive, this name
        reaches the generic OpenAI builder, which invents an entry carrying no
        retirement -- so a dead model would be counted and priced as a live one.
        """
        reloaded = user_registry("""
            [[model]]
            name = "GPT-Imaginary-9"
            provider = "openai"
            encoding = "o200k_base"
            retired = "2099-01-01"
        """)
        for spelling in ("GPT-Imaginary-9", "gpt-imaginary-9"):
            resolved = reloaded.get_model(spelling)
            assert resolved.name == "gpt-imaginary-9"
            assert resolved.retired == "2099-01-01"

    def test_a_capitalised_anthropic_name_keeps_its_tokenizer(self, user_registry):
        """Missing the registry entry would hand back a bare, untokenized model."""
        reloaded = user_registry("""
            [[model]]
            name = "Claude-Imaginary-9-20260101"
            provider = "anthropic"
            tokenizer = "claude-opus-4-7"
        """)
        for spelling in (
            "Claude-Imaginary-9-20260101",
            "claude-imaginary-9-20260101",
            "Claude-Imaginary-9",
            "claude-imaginary-9",
        ):
            resolved = reloaded.get_model(spelling)
            assert resolved.name == "claude-imaginary-9-20260101"
            assert resolved.tokenizer == reloaded.CLAUDE_TOKENIZER_OPUS_4_7

    @pytest.mark.parametrize(
        "overlay",
        [
            pytest.param("[[model]\nname = ", id="bad-syntax"),
            pytest.param(b'[[model]]\nname = "\xff\xfe-bad"\n', id="not-utf-8"),
        ],
    )
    def test_a_malformed_overlay_is_ignored_with_a_message(
        self, user_registry, capsys, overlay
    ):
        reloaded = user_registry(overlay)
        stderr = capsys.readouterr().err
        assert "models.toml" in stderr
        # The packaged registry still stands, so nothing the user typed can
        # leave toko unable to count.
        assert reloaded.get_model("claude-opus-5").provider == "anthropic"
        assert reloaded.list_models()["anthropic"]

    def test_overlay_models_stay_inside_their_tokenizer_generation(self, user_registry):
        """The Opus 4.7 guard must hold over registry data, not just literals."""
        reloaded = user_registry("""
            [[model]]
            name = "claude-imaginary-9-20260101"
            provider = "anthropic"
            tokenizer = "claude-legacy"

            [[model]]
            name = "claude-imaginary-9-20270101"
            provider = "anthropic"
            tokenizer = "claude-opus-4-7"
        """)
        for name, tokenizer in [
            ("claude-imaginary-9-20260101", reloaded.CLAUDE_TOKENIZER_LEGACY),
            ("claude-imaginary-9-20270101", reloaded.CLAUDE_TOKENIZER_OPUS_4_7),
        ]:
            for candidate in (name, f"{name}-latest"):
                resolved = reloaded.get_model(candidate)
                assert resolved.name == name
                assert resolved.tokenizer == tokenizer

        # The undated shorthand spans both generations, so it must resolve to
        # neither rather than silently pick one and report its token count.
        assert "claude-imaginary-9" not in reloaded._ANTHROPIC_ALIAS_MAP  # noqa: SLF001
        shorthand = reloaded.get_model("claude-imaginary-9")
        assert shorthand.name == "claude-imaginary-9"
        assert shorthand.tokenizer is None

    def test_overlay_models_in_one_generation_still_get_their_shorthand(
        self, user_registry
    ):
        reloaded = user_registry("""
            [[model]]
            name = "claude-imaginary-9-20260101"
            provider = "anthropic"
            tokenizer = "claude-legacy"

            [[model]]
            name = "claude-imaginary-9-20260202"
            provider = "anthropic"
            tokenizer = "claude-legacy"
        """)
        resolved = reloaded.get_model("claude-imaginary-9")
        assert resolved.name == "claude-imaginary-9-20260202"
        assert resolved.tokenizer == reloaded.CLAUDE_TOKENIZER_LEGACY
