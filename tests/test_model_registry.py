"""Tests for the TOML model registry and its user overlay."""

import importlib
import tomllib
from importlib import resources
from types import SimpleNamespace

import pytest

from toko import models


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
        grok_4 = models.XAI_MODELS["grok-4"]
        assert grok_4.retired == "2026-05-15"
        assert grok_4.redirects_to == "grok-4.3"
        assert grok_4.encoding == "o200k_base"

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
