"""Tests for the CLI."""

import ast
import json
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

import httpx
import pytest
import respx
from genai_prices.data_snapshot import set_custom_snapshot
from typer.testing import CliRunner

from tests.hf_hub import skip_if_rate_limited
from tests.pty_runner import HAS_PTY, PTY_SKIP_REASON, run_under_pty
from toko.cache import cache_count, get_cache_db_path, get_cached_count
from toko.cli import DEFAULT_JOBS, MAX_JOBS, _collect_supported_models, app
from toko.cost import format_cost, format_cost_value
from toko.counter import ANTHROPIC_COUNT_URL, GOOGLE_COUNT_URL_BASE, count_tokens
from toko.models import (
    _UNADVERTISED_LIVE_OPENAI_MODELS,
    RETIRED_OPENAI_MODELS,
    retirement_candidates,
    retirement_for_requested,
)
from toko.price_update import PRICE_DATA_URL, get_price_cache_path, get_price_data_path

runner = CliRunner()

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_BOX_DRAWING = re.compile(r"[─-╿]")


def _strip_ansi(text: str) -> str:
    """Rich splits styled help text with SGR codes whenever color is forced (as CI does)."""
    return _ANSI_ESCAPE.sub("", text)


def _normalize_cli_output(text: str) -> str:
    """Rich wraps panel prose to the terminal width, so drop borders and rejoin lines."""
    return " ".join(_BOX_DRAWING.sub(" ", _strip_ansi(text)).split())


@pytest.fixture(autouse=True)
def _isolated_config_home(tmp_path, monkeypatch):
    """Point config discovery at an empty per-test directory.

    CliRunner isolates the streams and the environment but never the filesystem, so
    the config lookup has to be pinned somewhere harmless from the outside.
    """
    config_home = tmp_path / "config"
    config_home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    return config_home


def _invoke_cli(
    args: list[str],
    env_overrides: dict[str, str] | None = None,
    stdin: str | None = None,
):
    """Invoke the CLI against the per-test config home."""
    return runner.invoke(app, args, env=env_overrides, input=stdin)


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip().startswith("toko version ")


SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "toko"


def _imports_click(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # `from .click import x` resolves inside toko, not to the click package.
            if node.level:
                continue
            names = [node.module or ""]
        else:
            continue
        if any(name.split(".")[0] == "click" for name in names):
            return True
    return False


def test_no_module_imports_click_directly():
    """Click is reached through typer, which vendored it in 0.26, and is not declared.

    Scanned rather than imported because whether the undeclared import fails is an
    accident of the resolution: with the [all] extras, huggingface-hub 1.x installs
    click whatever toko declares, and only a bare install breaks.
    """
    modules = sorted(SRC_ROOT.rglob("*.py"))
    offenders = [
        str(path.relative_to(SRC_ROOT))
        for path in modules
        if _imports_click(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
    # A scan that found nothing to read passes for the wrong reason.
    assert modules, f"no modules under {SRC_ROOT}"


# A tripwire that stops recognising the import is a tripwire that silently passes.
@pytest.mark.parametrize(
    ("source", "imported"),
    [
        ("import click", True),
        ("import os, click", True),
        ("import sys; import click", True),
        ("from click import Option", True),
        ("from click.core import Context", True),
        ('"""\nimport click\n"""', False),
        ("import typer\nfrom typer.core import TyperOption", False),
        ("import clicky\nfrom toko.clicker import thing", False),
        # Relative, so it names a toko submodule however much it reads like the package.
        ("from .click import Option", False),
        ("from ..click.core import Context", False),
    ],
)
def test_the_click_scan_reads_imports_not_text(source, imported):
    assert _imports_click(source) is imported


def test_list_models(monkeypatch):
    monkeypatch.setattr(
        "toko.cli.get_model_list",
        lambda **_kwargs: {
            "openai": ["gpt-4.1", "gpt-5"],
            "google": ["models/gemini-flash-latest"],
            "huggingface": ["meta-llama/Llama-3.2-1B"],
        },
    )

    result = _invoke_cli(["--list-models"])
    assert result.exit_code == 0
    lines = [line for line in result.stdout.splitlines() if line]
    assert lines == [
        "google/gemini-flash-latest",
        "meta-llama/Llama-3.2-1B",
        "openai/gpt-4.1",
        "openai/gpt-5",
    ]


def test_list_models_hides_retired_engines_without_the_flag():
    default = _invoke_cli(["--list-models"])
    with_retired = _invoke_cli(["--list-models", "--include-retired"])

    assert default.exit_code == 0
    assert with_retired.exit_code == 0
    assert "openai/text-davinci-003" not in default.stdout.splitlines()
    assert "openai/text-davinci-003" in with_retired.stdout.splitlines()
    assert "openai/gpt-4" in default.stdout.splitlines()


@pytest.mark.parametrize(
    "hidden", ["gpt-35-turbo", "gpt-3.5", "gpt2", "gpt-2", "babbage-002", "davinci-002"]
)
def test_a_name_that_counts_can_still_be_hidden_from_the_listing(hidden):
    """Hiding and refusing are separate decisions; these are hidden but not refused."""
    default = _invoke_cli(["--list-models"])
    with_retired = _invoke_cli(["--list-models", "--include-retired"])

    assert f"openai/{hidden}" not in default.stdout.splitlines()
    assert f"openai/{hidden}" in with_retired.stdout.splitlines()


def test_a_retired_openai_engine_is_refused_without_the_flag(tmp_path):
    """RETIRED_OPENAI_MODELS was silent before; naming one is now an error."""
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world")

    result = _invoke_cli(["--format", "csv", "-m", "text-davinci-003", str(sample)])

    assert result.exit_code == 1
    assert "model 'text-davinci-003' is retired (2024-01-04)." in result.stderr
    assert "--include-retired" in result.stderr
    assert result.stdout.strip() == ""


def test_a_retired_openai_engine_counts_when_opted_in(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world")

    result = _invoke_cli(
        ["--include-retired", "--format", "csv", "-m", "text-davinci-003", str(sample)]
    )

    assert result.exit_code == 0
    assert "sample.txt,2" in _strip_ansi(result.stdout)


def test_a_retired_registry_model_names_its_redirect_target():
    result = _invoke_cli(["-m", "grok-3", "--text", "hello world"])

    assert result.exit_code == 1
    assert result.stderr.splitlines()[0] == (
        "Error: model 'grok-3' is retired (2026-05-15); it redirects to grok-4.3. "
        "Pass --include-retired to count with it anyway."
    )


def test_a_retired_model_without_a_redirect_target_says_only_that_it_is_retired():
    result = _invoke_cli(["-m", "claude-3-opus-20240229", "--text", "hello world"])

    assert result.exit_code == 1
    error = result.stderr.splitlines()[0]
    assert error.startswith("Error: model 'claude-3-opus-20240229' is retired (2026-")
    assert "redirects to" not in error
    assert error.endswith("Pass --include-retired to count with it anyway.")


@pytest.mark.parametrize(
    ("requested", "retired"),
    [
        # The exact spelling --list-models --include-retired prints, which the README
        # sends users to grep.
        ("xai/grok-3", "grok-3"),
        ("openai/text-davinci-003", "text-davinci-003"),
        # Only Anthropic's resolver strips "-latest"; Google's leaves the alias for
        # countTokens and xAI's has no entry for it, so this used to resolve to a
        # fresh, unretired ModelInfo.
        ("grok-3-latest", "grok-3"),
    ],
)
def test_a_retired_model_is_refused_under_the_names_that_reach_it(requested, retired):
    result = _invoke_cli(["-m", requested, "--text", "hello world"])

    assert result.exit_code == 1
    assert result.stdout.strip() == ""
    assert result.stderr.splitlines()[0].startswith(
        f"Error: model '{requested}' is retired ("
    )
    assert retired in result.stderr


@pytest.mark.parametrize(
    "requested",
    [
        # The Azure deployment spelling of gpt-3.5-turbo, which does not shut down
        # until 2026-10-23. Refusing it while accepting gpt-3.5-turbo is arbitrary.
        "gpt-35-turbo",
        # A family shorthand, not an API name, but it resolves to the encoding the
        # live gpt-3.5-turbo family uses.
        "gpt-3.5",
        # Shutdown 2026-09-28, so still served today.
        "babbage-002",
        "davinci-002",
        # Open weights that were never OpenAI API models, so they never retired.
        "gpt2",
        "gpt-2",
    ],
)
def test_a_name_openai_still_serves_is_not_refused_as_retired(requested):
    """These are hidden from --list-models, which is not the same as being dead."""
    result = _invoke_cli(["-m", requested, "--text", "hello world"])

    assert result.exit_code == 0
    assert "retired" not in result.stderr
    assert result.stdout.strip() == "2"


# Every shut-down OpenAI engine and the date the gate must quote for it, written out
# rather than sampled: a parametrized handful let a silent edit to a date -- or a name
# quietly moved out of the mapping and back into the live set -- pass a green suite.
# cushman-codex and davinci-codex are None because they are /v1/engines-era names
# withdrawn before OpenAI published shutdown tables, so there is no date to quote.
SHUT_DOWN_OPENAI_ENGINES = {
    "ada": "2024-01-04",
    "babbage": "2024-01-04",
    "code-cushman-001": "2023-03-23",
    "code-cushman-002": "2023-03-23",
    "code-davinci-001": "2023-03-23",
    # Listed under two announcements: 2023-03-20 (Codex) with a 2023-03-23 shutdown,
    # and 2023-07-06 (GPT and embeddings) with a 2024-01-04 one. The later shutdown is
    # the one that happened, so the date here is neither announcement's own date.
    "code-davinci-002": "2024-01-04",
    "code-davinci-edit-001": "2024-01-04",
    "code-search-ada-code-001": "2024-01-04",
    "code-search-babbage-code-001": "2024-01-04",
    "curie": "2024-01-04",
    "cushman-codex": None,
    "davinci": "2024-01-04",
    "davinci-codex": None,
    "text-ada-001": "2024-01-04",
    "text-babbage-001": "2024-01-04",
    "text-curie-001": "2024-01-04",
    "text-davinci-001": "2024-01-04",
    "text-davinci-002": "2024-01-04",
    "text-davinci-003": "2024-01-04",
    "text-davinci-edit-001": "2024-01-04",
    "text-search-ada-doc-001": "2024-01-04",
    "text-search-babbage-doc-001": "2024-01-04",
    "text-search-curie-doc-001": "2024-01-04",
    "text-search-davinci-doc-001": "2024-01-04",
    "text-similarity-ada-001": "2024-01-04",
    "text-similarity-babbage-001": "2024-01-04",
    "text-similarity-curie-001": "2024-01-04",
    "text-similarity-davinci-001": "2024-01-04",
}


def test_the_shut_down_engine_table_is_exactly_the_one_the_gate_reads():
    """Membership is the gate's whole input, so an addition or removal has to show up."""
    assert dict(RETIRED_OPENAI_MODELS) == SHUT_DOWN_OPENAI_ENGINES


# Live tiktoken names --list-models hides, written out for the same reason the shut-down
# table above is: membership is the whole of the decision. Adding a name here drops a
# live model out of the default listing, and removing one starts advertising a name that
# was hidden on purpose -- neither shows up in any other assertion, so both were silent.
HIDDEN_LIVE_OPENAI_MODELS = {
    "babbage-002",
    "davinci-002",
    "gpt-2",
    "gpt-3.5",
    "gpt-35-turbo",
    "gpt2",
}


def test_the_hidden_live_name_set_is_exactly_the_one_the_listing_filters_on():
    """A name added here vanishes from --list-models; one removed appears in it."""
    assert set(_UNADVERTISED_LIVE_OPENAI_MODELS) == HIDDEN_LIVE_OPENAI_MODELS


@pytest.mark.parametrize(
    ("requested", "date"), sorted(SHUT_DOWN_OPENAI_ENGINES.items())
)
def test_a_shut_down_openai_engine_is_refused_with_its_recorded_date(requested, date):
    when = "date unknown" if date is None else date
    result = _invoke_cli(["-m", requested, "--text", "hello world"])

    assert result.exit_code == 1
    assert result.stdout.strip() == ""
    assert result.stderr.splitlines()[0] == (
        f"Error: model '{requested}' is retired ({when}). "
        "Pass --include-retired to count with it anyway."
    )


@pytest.mark.parametrize(
    "requested",
    [
        # A trailing space survived --model and left the gate matching nothing.
        "grok-3 ",
        " grok-3",
        # Only the first segment was stripped, so a router path kept the gate blind.
        "openrouter/xai/grok-3",
        "openrouter/xai/grok-3-latest",
    ],
)
def test_whitespace_and_multi_segment_prefixes_do_not_slip_past_the_gate(requested):
    result = _invoke_cli(["-m", requested, "--text", "hello world"])

    assert result.exit_code == 1
    assert result.stdout.strip() == ""
    assert f"model '{requested}' is retired (2026-05-15)" in result.stderr


@pytest.mark.parametrize(
    ("requested", "date"),
    [
        # A routing segment typed in the provider's own branding. Without the case fold
        # on each segment in _routed_model_name, "XAI" misses _ROUTING_PREFIX_SEGMENTS,
        # the name reads as a Hub repo owned by "XAI", and grok-3 counts with no
        # retirement notice at all -- exit 0 and a token count for a shut-down model.
        ("XAI/grok-3", "2026-05-15"),
        ("OpenRouter/xAI/grok-3", "2026-05-15"),
        # The "-latest" fold in retirement_candidates. Without it the alias is never
        # stripped, so the retired base name is never reached.
        ("grok-3-LATEST", "2026-05-15"),
        # The fold in retirement_of's RETIRED_OPENAI_MODELS lookup. detect_provider
        # already lowercases, so "CURIE" resolves to an OpenAI ModelInfo -- but the
        # ModelInfo keeps the name as typed, and the shut-down table is keyed lowercase.
        ("CURIE", "2024-01-04"),
        ("Text-Davinci-003", "2024-01-04"),
    ],
)
def test_a_retired_name_is_refused_however_it_is_capitalized(requested, date):
    """Three separate case folds stand between these spellings and a silent count.

    Every other prefixed or aliased case in the suite is spelled lowercase, so all three
    folds were unfenced: dropping any one of them turns the run from refused to counted
    and fails nothing in the suite outside the cases here -- the segment fold takes the
    two branded prefixes, the "-latest" fold takes grok-3-LATEST, and the retirement_of
    fold takes CURIE and Text-Davinci-003. A user types the branding ("XAI",
    "OpenRouter") or an old shouted engine name, not the lowercase spelling
    --list-models prints.
    """
    result = _invoke_cli(["-m", requested, "--text", "hello world"])

    assert result.exit_code == 1
    assert result.stdout.strip() == ""
    assert f"model '{requested}' is retired ({date})" in result.stderr


def test_a_bare_name_yields_exactly_one_candidate():
    """`not separator` and the dedupe are one guard between them, not two.

    Each is inert only while the other stands, so mutating either alone proves nothing.
    With no "/" in the name, rpartition leaves an empty prefix whose single empty segment
    reads as routing, so dropping the `not separator` check makes _routed_model_name
    return the whole name and retirement_candidates build it twice -- which dict.fromkeys
    then collapses. Drop the dedupe instead and nothing duplicates, because the check
    already returned None. Drop both and retirement_candidates("grok-3") is
    ["grok-3", "grok-3"]; the failures are this test and
    test_the_spelling_as_typed_decides_which_retirement_is_reported, whose asserted
    candidate list doubles the same way.
    """
    assert retirement_candidates("grok-3") == ["grok-3"]
    assert retirement_candidates("grok-3-latest") == ["grok-3-latest", "grok-3"]


@pytest.mark.parametrize(
    "requested",
    [
        # A real HuggingFace repo whose bare segment is a name OpenAI still serves.
        "openai-community/gpt2",
        # Hidden from --list-models but alive until 2026-09-28, prefixed or not.
        "openai/babbage-002",
    ],
)
def test_a_prefixed_name_that_is_not_retired_still_counts(requested):
    """Both tails are alive, so the gate clears these however the prefix is read.

    That makes this a check on the gate's verdict for a live name, not on the stripping
    rule -- the rule's two branches agree here, and the carve-outs are fenced by
    test_a_repo_owner_prefix_leaves_a_retired_tail_unread,
    test_a_google_prefix_leaves_a_retired_gemini_unread and
    test_a_huggingface_prefix_leaves_a_retired_tail_unread instead.

    The prefix is not inert outside the gate, and it costs both names the same thing.
    detect_provider splits the name once up front, and its first loop -- the one over
    tiktoken's model names -- is the only branch that reads that last segment alone;
    every branch below it reads the whole name, which is why google/gemini-2.5-pro
    resolves as huggingface and grok-lab/mymodel as xai. Both tails here are tiktoken
    model names ("gpt2" and "babbage-002"), so that first loop returns OpenAI and returns
    before the "/" branch further down can read either name as a Hub repo id. The counter
    then hands tiktoken the whole name, prefix included, which tiktoken has never seen,
    so both fall back to o200k_base and warn -- asserted below. That turns an exact count into a guess: on "Привет, мир! Как дела?",
    where the encodings disagree, bare babbage-002 counts 12 and bare gpt2 counts 23,
    while both spellings here count 8.

    Which of those two outcomes a prefixed name gets turns on the prefix's opening
    characters, not on its being a prefix. tiktoken.model.MODEL_PREFIX_TO_ENCODING holds
    17 entries and is matched with startswith, and only 5 of them are the ft: ones; a
    name whose owner begins with one of the other 12 -- o1-, o3-, o4-mini-, gpt-5-,
    gpt-4.5-, gpt-4.1-, chatgpt-4o-, gpt-4o-, gpt-4-, gpt-3.5-turbo-, gpt-35-turbo- or
    gpt-oss- -- resolves exactly and silently to that entry's encoding instead of
    falling back. "openai-community/" and "openai/" begin with none of the 12, which is
    why these two warn; gpt-4-lab/ does begin with one, and that case is fenced by
    test_an_owner_prefix_tiktoken_matches_picks_its_encoding_silently.
    """
    result = _invoke_cli(["-m", requested, "--text", "hello world"])

    assert result.exit_code == 0
    assert "retired" not in result.stderr
    assert (
        f"Warning: unknown OpenAI model '{requested}'; estimating with o200k_base"
        in result.stderr
    )


@pytest.mark.parametrize(
    ("requested", "expected_tokens"),
    [
        # gpt-4- -> cl100k_base, which counts this text differently from the o200k_base
        # fallback (12 against 8), so the number alone shows which one was used.
        ("gpt-4-lab/gpt2", 12),
        # gpt-oss- -> o200k_harmony, which agrees with the fallback at 8 here, so only
        # the silence and the exact flag separate the two paths.
        ("gpt-oss-lab/gpt2", 8),
        # o1- -> o200k_base. The gate read o1-labs/ as a repo owner and left the retired
        # tail unread, so a shut-down engine gets counted exactly under an encoding it
        # never used.
        ("o1-labs/text-davinci-003", 8),
    ],
)
def test_an_owner_prefix_tiktoken_matches_picks_its_encoding_silently(
    requested, expected_tokens
):
    """A prefix tiktoken matches costs an encoding, not a warning.

    _count_openai lowercases the requested name and alters nothing else about it -- all
    three names below are already lowercase, so what tiktoken sees is what was typed --
    and MODEL_PREFIX_TO_ENCODING is matched with startswith, so an owner whose text opens
    with one of tiktoken's 12 non-ft: model prefixes selects that entry's encoding with
    approximate=False and nothing on stderr. None of these names is an OpenAI model, and
    none of the encodings picked is the one the tail would have used: bare gpt2 counts 23
    under its own encoding rather than 12 or 8, and bare text-davinci-003 does not count
    at all because the retirement gate rejects it.
    """
    result = _invoke_cli(
        ["-m", requested, "--format", "json", "--text", "Привет, мир! Как дела?"]
    )

    assert result.exit_code == 0
    assert "Warning" not in result.stderr
    assert "retired" not in result.stderr
    count = json.loads(result.stdout)["totals"][0]
    assert count["tokens"] == expected_tokens
    assert count["approximate"] is False


@pytest.mark.parametrize(
    "requested",
    [
        # A leading slash left the prefix empty, which skipped the prefix check
        # entirely and let a shut-down engine count as an o200k_base guess. No Hub
        # repo can have an empty owner, so this names the engine and nothing else.
        "/text-davinci-003",
        # Router paths, which address a provider rather than owning a repo.
        "openrouter/text-davinci-003",
        "openrouter/openai/text-davinci-003",
        # anthropic/ is addressing too, so what is being named is OpenAI's curie. The
        # reviewer grounded this one specifically: no Hub repo owns the name.
        "anthropic/curie",
    ],
)
def test_a_routing_prefix_is_stripped(requested):
    """A prefix built only of provider names addresses a model; it does not own a repo."""
    result = _invoke_cli(["-m", requested, "--text", "hello world"])

    assert result.exit_code == 1
    assert result.stdout.strip() == ""
    assert f"model '{requested}' is retired (2024-01-04)" in result.stderr


# Live public Hugging Face repos whose last segment is an OpenAI engine's name. Each was
# confirmed against https://huggingface.co/api/models/<id> rather than assumed: calling
# any of them retired is a false statement about a repo that is still served. This is
# what the round-two rule got wrong -- it compared detect_provider on the whole name and
# on its last segment, which agrees for all 53 reachable retired names, so it stripped
# essentially always and refused Xenova/text-davinci-003 as "retired (2024-01-04)".
@pytest.mark.parametrize(
    "requested", ["Xenova/text-davinci-003", "openai-community/gpt2"]
)
def test_a_repo_owner_prefix_is_not_stripped(requested):
    """Only the Xenova case turns on the rule; gpt2 is live under either reading.

    Both are here because both have to keep counting, but a run is a coarse probe --
    it needs the repo to be served, so it cannot cover an owner whose retired tail is
    the interesting part. test_a_repo_owner_prefix_leaves_a_retired_tail_unread asks
    the gate directly and carries that half.
    """
    result = _invoke_cli(["-m", requested, "--text", "hello world"])

    assert result.exit_code == 0
    assert "retired" not in result.stderr


@pytest.mark.parametrize(
    "requested",
    [
        "Xenova/text-davinci-003",
        # An owner that is not a provider, on a tail that is. Whether this repo is
        # served was not checked and does not matter: the gate decides on the name
        # alone, and an owner it does not route for must leave the tail unread either
        # way. Counting it is a separate question this does not ask.
        "openai-community/text-davinci-003",
    ],
)
def test_a_repo_owner_prefix_leaves_a_retired_tail_unread(requested):
    """A rule that stripped a repo owner has to fail here.

    The tail is a shut-down engine, so stripping the owner reaches a retired name and
    the gate refuses. That is the only shape where the carve-out changes the answer,
    and before this the whole carve-out rested on the single Xenova run above.
    """
    assert retirement_candidates(requested) == [requested]
    assert retirement_for_requested(requested) is None


def test_a_google_repo_is_not_read_as_googles_own_model():
    """google/ owns real Hub repos, so it is a repo owner rather than addressing.

    google/gemma-3-1b-it is a real repo (google/gemma-2, which the round-two rule was
    documented against, is not). Asserted on the gate's verdict rather than through a
    run, because every google/gemma-* repo is gated and counting one needs Hub
    credentials the suite does not have. The gate is a pure name decision, so its
    verdict is the whole of its behaviour here.

    Both tails are live, so this says nothing about whether google/ is stripped -- it
    documents the repos the carve-out exists for. The carve-out itself is fenced by
    test_a_google_prefix_leaves_a_retired_gemini_unread.
    """
    assert retirement_for_requested("google/gemma-3-1b-it") is None
    assert retirement_for_requested("google/gemma-2-2b-it") is None


# The retired Gemini names --list-models prints under --include-retired. A retired tail
# is what makes the google/ carve-out observable: strip the prefix and the gate reads
# Google's own retired model and refuses, keep it and the name stays a Hub repo id.
#
# Read out of the listing rather than written out, so a sixth retired Gemini is fenced
# the moment the registry retires one. The spellings below pin today's set, the same way
# SHUT_DOWN_OPENAI_ENGINES pins the shut-down table: derivation alone would let the
# parametrization quietly empty out or widen.
def _retired_google_names_in_the_listing() -> list[str]:
    default = set(_collect_supported_models(include_retired=False))
    return [
        name
        for name in _collect_supported_models(include_retired=True)
        if name.startswith("google/") and name not in default
    ]


GOOGLE_RETIRED_LISTED_NAMES = _retired_google_names_in_the_listing()

RETIRED_GEMINI_LISTINGS = {
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3-pro-preview",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-2.0-flash-preview-image-generation",
}


def test_the_retired_gemini_listing_is_exactly_what_the_google_fence_covers():
    """A retired Google name added to or dropped from the registry has to show up.

    The fence below is parametrized over what the listing reports, so a registry change
    moves the fence with it -- and nothing would say so if the set changed by accident.
    """
    assert set(GOOGLE_RETIRED_LISTED_NAMES) == RETIRED_GEMINI_LISTINGS


@pytest.mark.parametrize("requested", GOOGLE_RETIRED_LISTED_NAMES)
def test_a_google_prefix_leaves_a_retired_gemini_unread(requested):
    """Dropping "google" from the exclusion set has to fail here.

    These are the spellings where the carve-out changes the answer: the bare name is
    retired, so a rule that stripped google/ would refuse them. Nothing else in the
    suite dies when google is removed from the exclusion, which is what makes this the
    fence rather than the gemma cases above.
    """
    assert retirement_candidates(requested) == [requested]
    assert retirement_for_requested(requested) is None


def test_a_huggingface_prefix_leaves_a_retired_tail_unread():
    """Dropping "huggingface" from the exclusion set has to fail here.

    The exclusion carves out two Hub organisations, and google was the only one fenced
    until this test: removing huggingface fails here and leaves the rest of the suite
    green, while the README went on promising the carve-out. huggingface/ owns real
    repos (huggingface/CodeBERTa-small-v1), so like google/ it is an owner, and a retired
    tail is the one shape that shows it -- strip the owner and the gate refuses a
    shut-down engine, keep it and the name stays a Hub repo id.
    """
    assert retirement_candidates("huggingface/text-davinci-003") == [
        "huggingface/text-davinci-003"
    ]
    assert retirement_for_requested("huggingface/text-davinci-003") is None


@pytest.mark.parametrize(
    "requested",
    [
        # A router path whose middle segment is a Hub owner rather than a provider.
        "openai/Xenova/text-davinci-003",
        "openrouter/Xenova/text-davinci-003",
    ],
)
def test_a_repo_owner_behind_a_routing_segment_still_blocks_the_strip(requested):
    """Every segment has to route, not just the first one.

    Narrowing _routed_model_name's `all(...)` to the leading segment fails both cases
    here and leaves the rest of the suite green -- every other multi-segment case routes
    the whole way down -- and flips `-m openai/Xenova/text-davinci-003` from a count to
    "retired (2024-01-04)".
    Xenova/text-davinci-003 is a live Hub repo, so a prefix ending in its owner is not
    pure addressing however it starts, and refusing it would be false about a repo that
    is still served. Asserted on the gate rather than a run: whether the Hub answers for
    the routed spelling is a separate question, and the gate decides on the name alone.
    """
    assert retirement_candidates(requested) == [requested]
    assert retirement_for_requested(requested) is None


def test_a_retirement_with_no_published_date_says_so():
    """RETIREMENT_DATE_UNKNOWN is the registry's stand-in, never a date to print.

    Collapsing retirement_of's `date=None if retired == RETIREMENT_DATE_UNKNOWN` to a
    plain `date=model_info.retired` fails this test and no other, turning this refusal
    into "is retired (unknown)" -- the sentinel leaking into the message as though
    "unknown" were the day xAI shut the model down. Three packaged entries carry it
    (grok-3-mini, grok-2-1212, grok-2-vision-1212) and nothing else asserts on the
    wording.
    """
    result = _invoke_cli(["-m", "grok-3-mini", "--text", "hello world"])

    assert result.exit_code == 1
    assert result.stdout.strip() == ""
    assert "model 'grok-3-mini' is retired (date unknown)" in result.stderr
    assert retirement_for_requested("grok-3-mini").date is None


def test_the_spelling_as_typed_decides_which_retirement_is_reported():
    """First non-None over the candidates, in the order retirement_candidates builds.

    The order is the whole of "as spelled": the typed name comes before its -latest
    strip, so a spelling that resolves on its own reports its own retirement. Reversing
    the loop in retirement_for_requested fails this test while moving the reported
    retirement onto the stripped base for every -latest spelling in the
    gemini-2.0-flash family -- 11 of them across the registry's names and Google's
    declared aliases -- and onto a different date for exactly one. This is that one:
    Google's -latest pass-through resolves the typed name to the gemini-2.0-flash-001
    family it would serve, retired 2026-06-01, while the suffix-stripped base is its own
    registry entry retired 2025-11-14. A silently different date is exactly what the gate
    exists to prevent.

    This sees the loop's order alone. The name below has no "/", so retirement_candidates
    builds the same list from it whichever order its `bases` are built in, and building
    them routed-first leaves this test green; that half is fenced by
    test_a_routed_spelling_that_is_its_own_entry_reports_its_own_retirement.
    """
    typed = "gemini-2.0-flash-preview-image-generation-latest"
    stripped = "gemini-2.0-flash-preview-image-generation"
    assert retirement_candidates(typed) == [typed, stripped]

    reported = retirement_for_requested(typed)
    assert reported.model == typed
    assert reported.date == "2026-06-01"
    assert retirement_for_requested(stripped).date == "2025-11-14"


def test_a_prefixed_retired_name_still_counts_when_opted_in():
    """--include-retired is the escape hatch for a name the gate now catches.

    The count it produces is the one this spelling always produced: the gate only
    refuses, it never resolves a name for counting.
    """
    result = _invoke_cli(
        ["--include-retired", "-m", "openai/text-davinci-003", "--text", "hello world"]
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "openai/text-davinci-003\t2\ttrue"


@pytest.mark.parametrize("requested", ["curie", "anthropic/curie"])
def test_the_opted_in_report_names_the_retirement_the_gate_refused_on(requested):
    """What the gate refuses on is what the run reports, whichever spelling was typed.

    The gate normalized the name and the reporter read it raw, so one run could refuse
    `anthropic/curie` as retired and then, once opted in, describe it as live:
    `"retirement": null` and no warning, while bare `curie` emitted the full object.
    Asserting only that the run counts is what let that pass, so both halves are pinned
    here, and both spellings have to produce the same retirement.
    """
    result = _invoke_cli(
        ["--include-retired", "-m", requested, "--format", "json", "--text", "hello"]
    )

    assert result.exit_code == 0
    assert (
        "Warning: curie was retired on 2024-01-04; the openai API will reject or "
        "redirect it." in result.stderr
    )
    payload = json.loads(result.stdout)
    assert payload["totals"][0]["retirement"] == {
        "model": "curie",
        "date": "2024-01-04",
        "redirects_to": None,
    }


def test_the_gate_leaves_a_live_provider_prefixed_name_resolving_as_it_did():
    """The gate only decides whether to refuse; it fixes none of get_model's gaps.

    openai/gpt-3.5-turbo still misses the registry and is estimated with o200k_base,
    exactly as it was before the gate learned to strip the prefix.
    """
    result = _invoke_cli(["-m", "openai/gpt-3.5-turbo", "--text", "hello world"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "openai/gpt-3.5-turbo\t2\ttrue"
    assert "unknown OpenAI model 'openai/gpt-3.5-turbo'" in result.stderr


def test_one_retired_model_fails_the_whole_run_before_anything_is_counted(tmp_path):
    """The run fails whole rather than printing a table with one column missing."""
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world")

    result = _invoke_cli(
        ["--format", "csv", "-m", "gpt-5", "-m", "grok-3", str(sample)]
    )

    assert result.exit_code == 1
    assert result.stdout.strip() == ""
    assert "grok-3" in result.stderr
    assert "gpt-5" not in result.stderr


def test_an_opted_in_retired_model_carries_its_retirement_into_json():
    """A shut-down engine tiktoken still carries, so this needs no key and no network.

    It used to ask for grok-3 and skip itself whenever the run failed, which meant a
    broken --include-retired -- the gate refusing what it just let through, say -- read
    as a skip rather than a failure. A non-null redirects_to is serialized in
    test_json_reports_a_retired_model_as_an_object, which needs no run at all.
    """
    result = _invoke_cli(
        [
            "--include-retired",
            "--format",
            "json",
            "-m",
            "text-davinci-003",
            "--text",
            "hi",
        ]
    )

    assert result.exit_code == 0
    count = _only_count(result)
    assert count["tokens"] == 1
    assert count["retirement"] == {
        "model": "text-davinci-003",
        "date": "2024-01-04",
        "redirects_to": None,
    }


@pytest.mark.slow
def test_update_prices_subcommand_is_reached():
    result = runner.invoke(app, ["update-prices"])
    assert result.exit_code == 0
    assert "Successfully updated pricing data" in result.stdout


@respx.mock
def test_update_prices_reports_download_failure():
    respx.get(PRICE_DATA_URL).mock(return_value=httpx.Response(503))

    result = runner.invoke(app, ["update-prices"])

    assert result.exit_code == 1
    assert "Failed to fetch pricing data" in result.stderr
    assert "Successfully updated" not in result.stdout


def test_clear_cache_subcommand_is_reached():
    result = runner.invoke(app, ["clear-cache"])
    assert result.exit_code == 0
    assert "Cache cleared" in result.stdout


def test_clear_cache_removes_the_price_cache():
    """Clearing the cache must reach the pricing payload, not just the token database."""
    get_price_data_path().write_bytes(b"not json")
    get_price_cache_path().write_text("0")

    result = runner.invoke(app, ["clear-cache"])

    assert result.exit_code == 0
    assert not get_price_data_path().exists()
    assert not get_price_cache_path().exists()


# Each subcommand echoes its marker before doing any work, so the marker proves the
# name was dispatched even when the work behind it fails.
@pytest.mark.parametrize(
    ("args", "marker"),
    [
        # The bare form is the control: it survives even a broken option scan, so the
        # separated forms below are the ones that actually prove the scan works.
        (["clear-cache"], "Cache cleared"),
        (["--total-only", "clear-cache"], "Cache cleared"),
        (["-m", "gpt-5", "clear-cache"], "Cache cleared"),
        (["--format", "csv", "clear-cache"], "Cache cleared"),
        (["--sort", "count", "clear-cache"], "Cache cleared"),
        # Glued forms consume no following token, so they survive it too.
        (["--format=csv", "clear-cache"], "Cache cleared"),
        (["-mgpt-5", "clear-cache"], "Cache cleared"),
    ],
)
def test_global_option_before_a_subcommand_still_dispatches(args, marker):
    """A leading global option must not hide the subcommand name behind PATHS."""
    result = _invoke_cli(args)

    output = _strip_ansi(result.stdout)
    assert "Path not found" not in output + _strip_ansi(result.stderr)
    assert marker in output


@respx.mock
def test_global_option_before_update_prices_still_dispatches():
    """The other subcommand, dispatched behind a global option without the network."""
    respx.get(PRICE_DATA_URL).mock(return_value=httpx.Response(503))

    result = _invoke_cli(["-m", "gpt-5", "update-prices"])

    assert "Path not found" not in _strip_ansi(result.stdout + result.stderr)
    assert "Fetching latest pricing data" in _strip_ansi(result.stdout)


def test_subcommand_name_as_an_option_value_is_not_dispatched(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world")

    result = _invoke_cli(["-m", "clear-cache", str(sample)])

    assert "Cache cleared" not in _strip_ansi(result.stdout)
    assert "clear-cache" in _strip_ansi(result.stderr)


def test_subcommand_name_after_a_double_dash_is_a_path():
    result = _invoke_cli(["--", "clear-cache"])

    assert result.exit_code == 1
    assert "Cache cleared" not in _strip_ansi(result.stdout)
    assert "Path not found: clear-cache" in _strip_ansi(result.stderr)


@pytest.mark.parametrize("subcommand", ["update-prices", "clear-cache"])
def test_subcommand_help_describes_the_subcommand(subcommand):
    """`toko <sub> --help` must not fall through to the top-level help."""
    result = runner.invoke(app, [subcommand, "--help"])

    assert result.exit_code == 0
    output = _strip_ansi(result.stdout)
    assert f"Usage: toko {subcommand} [OPTIONS]" in output
    assert "--list-models" not in output


# One provider priced far away from anything genai-prices bundles, so a cost that
# matches it can only have come from the payload this test served.
_SENTINEL_PRICES = [
    {
        "id": "openai",
        "name": "OpenAI",
        "api_pattern": r"api\.openai\.com",
        "models": [
            {
                "id": "gpt-5",
                "name": "GPT-5",
                "match": {"equals": "gpt-5"},
                "prices": {"input_mtok": 999.0, "output_mtok": 999.0},
            }
        ],
    }
]


@respx.mock
def test_counting_run_uses_prices_from_an_earlier_update():
    """`update-prices` must reach later runs even with auto-update off (the default)."""
    respx.get(PRICE_DATA_URL).mock(
        return_value=httpx.Response(200, json=_SENTINEL_PRICES)
    )
    env = {"TOKO_AUTO_UPDATE_PRICES": "false"}

    assert _invoke_cli(["update-prices"], env).exit_code == 0

    # Drop the in-process snapshot so the count below starts where a fresh process
    # would: on the bundled prices, with only prices.json to recover the update from.
    set_custom_snapshot(None)

    result = _invoke_cli(
        [
            "--cost",
            "--header",
            "--format",
            "tsv",
            "-m",
            "gpt-5",
            "--text",
            "hello world",
        ],
        env,
    )

    assert result.exit_code == 0
    # 2 tokens at the sentinel $999/Mtok; the bundled price would be near $0.000003.
    # TSV carries the cost as a bare number, so read it as one.
    assert float(result.stdout.splitlines()[1].split("\t")[2]) == pytest.approx(
        0.001998
    )
    assert respx.calls.call_count == 1


def test_count_with_text():
    result = _invoke_cli(["--header", "--format", "tsv", "--text", "hello world"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "model\ttokens\ngpt-5\t2"


def test_count_with_text_default_output():
    result = _invoke_cli(["--text", "hello world"])
    assert result.exit_code == 0
    expected = count_tokens("hello world", model="gpt-5")
    assert result.stdout.strip() == str(expected.count)


def test_count_from_stdin():
    result = _invoke_cli(["--header", "--format", "tsv"], stdin="hello world")
    assert result.exit_code == 0
    assert result.stdout.strip() == "model\ttokens\ngpt-5\t2"


def test_count_with_multiple_models():
    result = _invoke_cli(
        [
            "--header",
            "--format",
            "tsv",
            "--model",
            "gpt-5",
            "--model",
            "gpt-5-mini",
            "--text",
            "hello",
        ]
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "model\ttokens\ngpt-5\t1\ngpt-5-mini\t1"


def test_cost_column_in_tsv():
    result = _invoke_cli(
        [
            "--header",
            "--format",
            "tsv",
            "--model",
            "gpt-5",
            "--model",
            "gpt-4.1",
            "--text",
            "hello",
            "--cost",
        ]
    )
    lines = [line for line in result.stdout.splitlines() if line]
    assert lines[0] == "model\ttokens\tcost"
    assert lines[1].startswith("gpt-5\t")


_COUNT_KEYS = {"model", "tokens", "approximate", "cost", "caveats", "retirement"}


def _envelope(result) -> dict:
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    return payload


def _only_count(result) -> dict:
    """Return the one count of a single-model run, checking the invariant keys."""
    payload = _envelope(result)
    counts = [count for entry in payload["results"] for count in entry["counts"]]
    for count in [*counts, *payload["totals"]]:
        assert set(count) == _COUNT_KEYS
    assert len(counts) == 1
    return counts[0]


def test_json_wraps_a_text_run_in_the_envelope():
    result = _invoke_cli(["--format", "json", "--model", "gpt-5", "--text", "hello"])
    assert result.exit_code == 0
    assert _envelope(result) == {
        "schema_version": 1,
        "results": [
            {
                "source": {"kind": "text", "name": None},
                "counts": [
                    {
                        "model": "gpt-5",
                        "tokens": 1,
                        "approximate": False,
                        "cost": None,
                        "caveats": [],
                        "retirement": None,
                    }
                ],
            }
        ],
        "totals": [
            {
                "model": "gpt-5",
                "tokens": 1,
                "approximate": False,
                "cost": None,
                "caveats": [],
                "retirement": None,
            }
        ],
    }


def test_json_reports_a_null_cost_without_the_flag():
    """The key is there either way; only the value says whether costing was asked for."""
    result = _invoke_cli(["--format", "json", "--model", "gpt-5", "--text", "hello"])
    assert result.exit_code == 0
    assert _only_count(result)["cost"] is None


def test_json_includes_cost_with_flag():
    result = _invoke_cli(
        ["--format", "json", "--model", "gpt-5", "--text", "hello", "--cost"]
    )
    assert result.exit_code == 0
    count = _only_count(result)
    assert count["tokens"] == 1
    assert count["cost"] > 0


def test_json_keys_do_not_move_with_the_cost_flag():
    """The whole point for a jq consumer: no key appears or vanishes on a flag."""
    without = _invoke_cli(["--format", "json", "-m", "gpt-5", "--text", "hello world"])
    with_cost = _invoke_cli(
        ["--format", "json", "--cost", "-m", "gpt-5", "--text", "hello world"]
    )
    assert set(_only_count(without)) == set(_only_count(with_cost)) == _COUNT_KEYS


def test_json_file_output_includes_cost_with_flag(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world")

    result = _invoke_cli(
        ["--format", "json", "--model", "gpt-5", "--cost", str(sample)]
    )
    assert result.exit_code == 0
    count = _only_count(result)
    assert count["tokens"] == 2
    assert count["cost"] > 0


def test_json_names_the_source_a_text_run_and_a_file_run_came_from(tmp_path):
    """A text run and a file run were indistinguishable before the envelope."""
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world")

    from_text = _invoke_cli(["--format", "json", "-m", "gpt-5", "-t", "hello world"])
    from_file = _invoke_cli(["--format", "json", "-m", "gpt-5", str(sample)])

    assert _envelope(from_text)["results"][0]["source"] == {
        "kind": "text",
        "name": None,
    }
    assert _envelope(from_file)["results"][0]["source"] == {
        "kind": "file",
        "name": str(sample),
    }


def test_json_reports_an_approximate_count_and_the_reason_for_it():
    result = _invoke_cli(["--format", "json", "-m", "gpt-6", "--text", "hello world"])
    assert result.exit_code == 0
    count = _only_count(result)
    assert count["tokens"] == 2
    assert count["approximate"] is True
    assert count["caveats"] == [
        {
            "kind": "openai_encoding_guess",
            "model": "gpt-6",
            "message": "unknown OpenAI model 'gpt-6'; estimating with o200k_base",
            "encoding": "o200k_base",
            "tokenizer": None,
            "reason": None,
        }
    ]


def test_json_gives_an_exact_and_an_approximate_count_the_same_keys():
    """One document must never mix shapes."""
    result = _invoke_cli(
        ["--format", "json", "-m", "gpt-5", "-m", "gpt-6", "--text", "hello world"]
    )
    assert result.exit_code == 0
    counts = _envelope(result)["results"][0]["counts"]
    assert [set(count) for count in counts] == [_COUNT_KEYS, _COUNT_KEYS]
    assert [count["approximate"] for count in counts] == [False, True]
    assert counts[0]["caveats"] == []


def _caveat_of(count: dict) -> dict:
    assert len(count["caveats"]) == 1
    return count["caveats"][0]


def test_a_caveat_holding_a_semicolon_survives_a_merged_total(tmp_path):
    """The real unknown-OpenAI message contains "; ", which used to be the joiner.

    Two files merge into one total, and the caveat has to arrive whole: joining the
    per-file caveats into one string made the total's punctuation ambiguous, so a
    consumer splitting it back apart got four fragments out of two caveats.
    """
    for name in ("first.txt", "second.txt"):
        (tmp_path / name).write_text("hello world")

    result = _invoke_cli(
        ["--format", "json", "--total-only", "-m", "gpt-9-turbo", str(tmp_path)]
    )

    assert result.exit_code == 0
    (total,) = _envelope(result)["totals"]
    assert total["tokens"] == 4
    assert _caveat_of(total) == {
        "kind": "openai_encoding_guess",
        "model": "gpt-9-turbo",
        "message": "unknown OpenAI model 'gpt-9-turbo'; estimating with o200k_base",
        "encoding": "o200k_base",
        "tokenizer": None,
        "reason": None,
    }


def test_distinct_caveats_across_models_merge_without_losing_or_crossing(tmp_path):
    for name in ("first.txt", "second.txt"):
        (tmp_path / name).write_text("hello world")

    result = _invoke_cli(
        [
            "--format",
            "json",
            "--total-only",
            "-m",
            "gpt-9-turbo",
            "-m",
            "gpt-8-turbo",
            "-m",
            "gpt-5",
            str(tmp_path),
        ]
    )

    assert result.exit_code == 0
    totals = {total["model"]: total for total in _envelope(result)["totals"]}
    # One caveat per column even though two files contributed it: identical caveats
    # collapse rather than piling up once per file.
    assert _caveat_of(totals["gpt-9-turbo"])["model"] == "gpt-9-turbo"
    assert _caveat_of(totals["gpt-8-turbo"])["model"] == "gpt-8-turbo"
    # An exact model in the same run is not tarred with its neighbours' caveats.
    assert totals["gpt-5"]["caveats"] == []


def test_json_lists_one_model_order_across_a_multi_model_multi_file_run(tmp_path):
    """The case that was untested: several models over several files.

    What is guaranteed is the order, not the length: a model a source could not be
    counted for is absent from that source's array, so a reader matches on `model`.
    Two orders in one document would defeat even that, since the models a reader
    scans for would appear in a different sequence per array.
    """
    args = _two_file_args(tmp_path)
    result = _invoke_cli(
        ["--format", "json", "-m", "gpt-5", "-m", "gpt-4o-mini", *args]
    )

    assert result.exit_code == 0
    payload = _envelope(result)
    orders = [
        [count["model"] for count in counts]
        for counts in [
            *(result_entry["counts"] for result_entry in payload["results"]),
            payload["totals"],
        ]
    ]
    # The order the models were named, everywhere in the document.
    assert orders == [["gpt-5", "gpt-4o-mini"]] * 3


def test_csv_and_tsv_costs_are_bare_numbers_and_text_keeps_the_dollar_sign(monkeypatch):
    args = ["--cost", "--header", "-m", "gpt-5", "--text", "hello world"]
    csv_run = _invoke_cli(["--format", "csv", *args])
    tsv_run = _invoke_cli(["--format", "tsv", *args])

    assert csv_run.exit_code == 0
    assert tsv_run.exit_code == 0
    csv_cost = csv_run.stdout.splitlines()[1].split(",")[2]
    tsv_cost = tsv_run.stdout.splitlines()[1].split("\t")[2]
    assert "$" not in csv_cost
    assert float(csv_cost) == float(tsv_cost) > 0

    # The text table is the one people read, and it keeps the currency.
    monkeypatch.setattr("toko.cli.is_stdout_tty", lambda: True)
    text_run = _invoke_cli(args)
    assert text_run.exit_code == 0
    assert "$" in _strip_ansi(text_run.stdout)


def test_a_fraction_of_a_cent_keeps_the_precision_the_display_format_drops(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hi")

    result = _invoke_cli(
        ["--cost", "--header", "--format", "csv", "-m", "gpt-5", str(sample)]
    )

    assert result.exit_code == 0
    cell = result.stdout.splitlines()[1].split(",")[2]
    cost = float(cell)
    assert cost > 0
    # The human format pads to six decimals and loses everything below them, which
    # is why the machine-readable column cannot borrow it.
    assert format_cost(cost) == "$0.000001"
    assert cost != float(format_cost(cost).lstrip("$"))


@pytest.mark.parametrize(
    "cost", [1234.5678, 98765.4321, 0.000123456789, 1e-06, 5.5e-05]
)
def test_delimited_cost_cell_round_trips_without_rounding(cost):
    """A total too large for six significant figures still has to survive the cell.

    Reached directly rather than through the CLI: a run that really costs $1234
    is not something a test can manufacture.
    """
    assert float(format_cost_value(cost)) == cost


def test_delimited_cost_cell_suppresses_float_accumulation_noise():
    assert format_cost_value(0.1 + 0.2) == "0.3"


@pytest.mark.parametrize(
    "cost", [1234.5678, 98765.4321, 0.000123456789, 1e-06, 5.5e-05, 9.9e-09]
)
def test_delimited_cost_cell_is_a_positional_decimal(cost):
    """`sort -n` reads 3.75e-06 as 3.75, and `bc` refuses it outright."""
    cell = format_cost_value(cost)
    assert "e" not in cell.lower()
    assert float(cell) == cost


def _sorted_by_sort_n(cells: list[str]) -> list[str]:
    sort = shutil.which("sort")
    assert sort is not None
    completed = subprocess.run(  # noqa: S603
        [sort, "-n"],
        input="\n".join(cells),
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    return completed.stdout.split()


@pytest.mark.skipif(shutil.which("sort") is None, reason="needs a POSIX sort")
def test_sort_n_orders_a_cost_column_spanning_orders_of_magnitude():
    costs = [1234.5678, 0.5, 0.000125, 3.75e-06, 5e-06, 9.9e-09]
    cells = [format_cost_value(cost) for cost in costs]

    assert _sorted_by_sort_n(cells) == [
        format_cost_value(cost) for cost in sorted(costs)
    ]


def _delimited_costs(stdout: str, separator: str) -> dict[str, float]:
    header, *rows = [line.split(separator) for line in stdout.splitlines() if line]
    column = header.index("gpt-5_cost")
    return {row[0]: float(row[column]) for row in rows}


def test_every_format_reports_one_cost_for_the_same_run(tmp_path):
    """JSON used to leak the float-accumulation noise the delimited cell had shed.

    Two files, so the totals are a sum rather than a copy, and a pair of token counts
    whose costs do not sum cleanly: three and four gpt-5 tokens price at 3.75e-06 and
    5e-06, whose float sum is 8.750000000000001e-06. Without the fix JSON prints that,
    beside a CSV reading 0.00000875. A pair that happens to sum exactly -- two and
    three tokens, say -- passes whether the fix is there or not, so it is not a test.
    """
    first = tmp_path / "first.txt"
    first.write_text("hello wider world")
    second = tmp_path / "second.txt"
    second.write_text("hello a wider world")
    args = ["--cost", "-m", "gpt-5", str(first), str(second)]

    payload = _envelope(_invoke_cli(["--format", "json", *args]))
    per_file = {
        entry["source"]["name"]: entry["counts"][0]["cost"]
        for entry in payload["results"]
    }
    (total,) = payload["totals"]

    # Spelled out so an edit to the fixture that loses the inexact sum is visible here
    # rather than quietly turning the assertions below into a tautology.
    assert sorted(per_file.values()) == [3.75e-06, 5e-06]
    assert total["cost"] == 8.75e-06

    for separator, name in ((",", "csv"), ("\t", "tsv")):
        rows = _invoke_cli(["--format", name, "--header", *args])
        assert rows.exit_code == 0
        assert _delimited_costs(rows.stdout, separator) == per_file

        summed = _invoke_cli(["--format", name, "--header", "--total-only", *args])
        assert summed.exit_code == 0
        assert _delimited_costs(summed.stdout, separator) == {"TOTAL": total["cost"]}

    # Every cost the run reports is already at the precision the cell renders, so no
    # format has anything left to round off on its own.
    assert all(
        cost == float(f"{cost:.12g}") for cost in [*per_file.values(), total["cost"]]
    )


def test_csv_gains_the_approximate_column_only_when_a_count_is_approximate():
    exact = _invoke_cli(
        ["--header", "--format", "csv", "-m", "gpt-5", "--text", "hello world"]
    )
    approximate = _invoke_cli(
        [
            "--header",
            "--format",
            "csv",
            "-m",
            "gpt-5",
            "-m",
            "gpt-6",
            "--text",
            "hello world",
        ]
    )

    assert exact.stdout.strip() == "model,tokens\ngpt-5,2"
    assert approximate.stdout.strip() == (
        "model,tokens,approximate\ngpt-5,2,false\ngpt-6,2,true"
    )


def test_piped_single_model_tsv_keeps_approximate_marker():
    # Without a header there is nowhere else for the marker to go, and stderr is
    # not where a consumer on the other end of the pipe is looking.
    result = _invoke_cli(
        ["--format", "tsv", "--model", "gpt-6", "--text", "hello world"]
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "gpt-6\t2\ttrue"
    assert "unknown OpenAI model 'gpt-6'" in result.stderr


def test_piped_single_model_tsv_stays_bare_when_exact():
    result = _invoke_cli(
        ["--format", "tsv", "--model", "gpt-5", "--text", "hello world"]
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "2"


def test_default_no_header_when_not_tty(monkeypatch):
    monkeypatch.setattr("toko.cli.is_stdout_tty", lambda: False)
    text = "header-default-pipe"
    result = _invoke_cli(["--model", "gpt-5", "--model", "gpt-5-mini", "--text", text])
    lines = [line for line in result.stdout.splitlines() if line]
    assert lines
    assert lines[0].startswith("gpt-5\t")
    assert lines[1].startswith("gpt-5-mini\t")


def test_header_flag_forces_header(monkeypatch):
    monkeypatch.setattr("toko.cli.is_stdout_tty", lambda: False)
    text = "header-flag"
    result = _invoke_cli(
        ["--header", "--model", "gpt-5", "--model", "gpt-5-mini", "--text", text]
    )
    first_line = next((line for line in result.stdout.splitlines() if line), "")
    assert first_line.lower().startswith("model")


def test_tty_default_includes_header(monkeypatch):
    monkeypatch.setattr("toko.cli.is_stdout_tty", lambda: True)
    text = "tty-header"
    result = _invoke_cli(
        ["--format", "tsv", "--model", "gpt-5", "--model", "gpt-5-mini", "--text", text]
    )
    first_line = next((line for line in result.stdout.splitlines() if line), "")
    assert first_line.lower().startswith("model")
    assert any(line.startswith("gpt-5\t") for line in result.stdout.splitlines())


def test_no_header_flag_respected(monkeypatch):
    monkeypatch.setattr("toko.cli.is_stdout_tty", lambda: True)
    text = "tty-no-header"
    result = _invoke_cli(
        [
            "--no-header",
            "--format",
            "tsv",
            "--model",
            "gpt-5",
            "--model",
            "gpt-5-mini",
            "--text",
            text,
        ]
    )
    first_line = next((line for line in result.stdout.splitlines() if line), "")
    assert not first_line.lower().startswith("model")


def test_partial_success_missing_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not os.environ.get("ANTHROPIC_API_KEY")

    text = "test-missing-anthropic-partial"
    result = _invoke_cli(
        ["--model", "gpt-5", "--model", "claude-sonnet-4-5", "--text", text]
    )
    assert result.exit_code == 0
    expected = count_tokens(text, model="gpt-5")
    assert result.stdout.strip() == str(expected.count)
    assert "claude-sonnet-4-5" not in result.stdout
    assert "Failed to count tokens for claude-sonnet-4-5" in result.stderr
    assert "ANTHROPIC_API_KEY" in result.stderr


def test_all_fail_missing_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not os.environ.get("ANTHROPIC_API_KEY")

    text = "test-missing-anthropic-all"
    result = _invoke_cli(["--model", "claude-sonnet-4-5", "--text", text])
    assert result.exit_code != 0
    assert "Error: All models failed to count tokens" in result.stderr


def test_partial_success_missing_google_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert not os.environ.get("GOOGLE_API_KEY")

    text = "test-missing-google-partial"
    result = _invoke_cli(
        ["--model", "gpt-5", "--model", "models/gemini-2.5-flash", "--text", text]
    )
    assert result.exit_code == 0
    expected = count_tokens(text, model="gpt-5")
    assert result.stdout.strip() == str(expected.count)
    assert "gemini-2.5-flash" not in result.stdout
    assert "GOOGLE_API_KEY" in result.stderr


@respx.mock
def test_anthropic_bad_response_reports_error_without_traceback():
    respx.post(ANTHROPIC_COUNT_URL).mock(
        return_value=httpx.Response(200, json={"unexpected": "shape"})
    )

    result = _invoke_cli(
        ["--model", "claude-sonnet-4-5", "--text", "hello"],
        {"ANTHROPIC_API_KEY": "test-key"},
    )

    assert isinstance(result.exception, SystemExit)
    assert result.exit_code == 1
    assert "Unexpected response from Anthropic" in result.stderr
    assert "Error: All models failed to count tokens" in result.stderr


@respx.mock
def test_google_bad_response_reports_error_without_traceback():
    respx.post(url__startswith=GOOGLE_COUNT_URL_BASE).mock(
        return_value=httpx.Response(200, json={"unexpected": "shape"})
    )

    result = _invoke_cli(
        ["--model", "gemini-2.5-flash", "--text", "hello"],
        {"GOOGLE_API_KEY": "test-key"},
    )

    assert isinstance(result.exception, SystemExit)
    assert result.exit_code == 1
    assert "Unexpected response from Google" in result.stderr
    assert "Error: All models failed to count tokens" in result.stderr


def test_option_after_positional_path(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world")

    result = _invoke_cli([str(sample), "--total-only", "--format", "csv"])
    assert result.exit_code == 0
    assert result.stdout.strip().endswith("TOTAL,2")


def test_short_option_after_positional_path(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world")

    result = _invoke_cli([str(sample), "-m", "gpt-5", "--format", "tsv"])
    assert result.exit_code == 0
    assert "\t2" in result.stdout


def test_help_after_positional_path(tmp_path):
    result = _invoke_cli([str(tmp_path), "--help"])
    assert result.exit_code == 0
    assert "Usage: toko" in _normalize_cli_output(result.stdout)


def test_bad_path_does_not_abort_good_paths(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world")
    missing = tmp_path / "nope.txt"

    result = _invoke_cli(["--format", "csv", str(sample), str(missing)])
    assert result.exit_code == 1
    assert "sample.txt,2" in result.stdout
    assert "nope.txt" in result.stderr


# The .invalid TLD never resolves, so any of these that escaped respx would fail
# loudly instead of quietly reaching the network.
_GOOD_URL = "https://toko.invalid/good.txt"
_BAD_URL = "https://toko.invalid/bad.txt"


@respx.mock
def test_url_path_counts_tokens():
    respx.get(_GOOD_URL).mock(return_value=httpx.Response(200, text="hello world"))

    result = _invoke_cli(["--format", "csv", "-m", "gpt-5", _GOOD_URL])
    assert result.exit_code == 0
    assert f"{_GOOD_URL},2" in result.stdout


@respx.mock
def test_unfetchable_url_reports_an_error():
    respx.get(_BAD_URL).mock(return_value=httpx.Response(404))

    result = _invoke_cli(["--format", "csv", "-m", "gpt-5", _BAD_URL])
    assert result.exit_code == 1
    assert f"Error fetching URL {_BAD_URL}" in result.stderr
    assert result.stdout.strip() == ""


@respx.mock
def test_bad_url_does_not_abort_good_url_or_file(tmp_path):
    respx.get(_GOOD_URL).mock(return_value=httpx.Response(200, text="hello world"))
    respx.get(_BAD_URL).mock(side_effect=httpx.ConnectError("boom"))
    sample = tmp_path / "sample.txt"
    sample.write_text("goodbye world friend")

    result = _invoke_cli(
        ["--format", "csv", "-m", "gpt-5", _GOOD_URL, _BAD_URL, str(sample)]
    )
    assert result.exit_code == 1
    assert f"{_GOOD_URL},2" in result.stdout
    assert "sample.txt,4" in result.stdout
    assert f"Error fetching URL {_BAD_URL}" in result.stderr
    assert _BAD_URL not in result.stdout


def test_unreadable_file_in_directory_does_not_abort_batch(tmp_path):
    (tmp_path / "sample.txt").write_text("hello world")
    # A self-referential symlink is unreadable for every user, including root.
    (tmp_path / "loop.txt").symlink_to("loop.txt")

    result = _invoke_cli(["--format", "csv", str(tmp_path)])
    assert result.exit_code == 1
    assert "sample.txt,2" in result.stdout
    assert "Error reading" in result.stderr
    assert "loop.txt" in result.stderr


def test_missing_path_reports_only_the_specific_error(tmp_path):
    result = _invoke_cli([str(tmp_path / "nope.txt")])
    assert result.exit_code == 1
    assert "Error: Path not found" in result.stderr
    assert "No files found matching criteria" not in result.stderr


def test_empty_directory_reports_no_files(tmp_path):
    result = _invoke_cli([str(tmp_path)])
    assert result.exit_code == 1
    assert "Error: No files found matching criteria" in result.stderr


def test_redirected_empty_stdin_is_input_and_counts_to_zero():
    """`toko -m gpt-5 < /dev/null` exits 0 printing 0; the guard is about the terminal."""
    result = _invoke_cli(["-m", "gpt-5"], stdin="")
    assert result.exit_code == 0
    assert result.stdout.strip() == "0"
    assert "No input provided" not in result.stderr


@pytest.mark.skipif(not HAS_PTY, reason=PTY_SKIP_REASON)
def test_the_no_input_error_needs_a_terminal_on_stdin(tmp_path):
    """The other half of the pair above, which only a real tty can reach.

    stdout is a pipe here so the exit code can be read back; stdin stays the terminal,
    which is the condition being tested.
    """
    script = tmp_path / "tty_stdin_driver.py"
    script.write_text(
        "import sys\n\n"
        "from toko.cli import app\n\n"
        "assert sys.stdin.isatty()\n"
        "try:\n"
        "    app(['-m', 'gpt-5'])\n"
        "except SystemExit as exit_status:\n"
        "    print(f'EXIT={exit_status.code}')\n"
    )

    output = run_under_pty(str(script), dict(os.environ), pipe_stdout=True)

    assert "EXIT=1" in output


def test_a_binary_file_among_good_paths_is_skipped_without_failing_the_run(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("hello world")
    binary = tmp_path / "image.bin"
    binary.write_bytes(b"\x00\x01\x02\xff\xfe")

    result = _invoke_cli(["--format", "csv", "-m", "gpt-5", str(good), str(binary)])

    assert result.exit_code == 0
    assert f"Warning: Skipping binary file {binary}" in result.stderr
    # Skipped means absent, not zero: it contributes no row and no tokens.
    assert binary.name not in result.stdout
    assert result.stdout.splitlines() == [f"{good},2"]


def test_a_partial_failure_still_emits_a_whole_envelope_with_short_totals(tmp_path):
    """Exit 1 is the only signal that a total covers less than it was asked for."""
    args = _two_file_args(tmp_path)
    missing = str(tmp_path / "nope.txt")

    partial = _invoke_cli(["--format", "json", "-m", "gpt-5", *args, missing])
    whole = _invoke_cli(["--format", "json", "-m", "gpt-5", *args])

    assert partial.exit_code == 1
    assert whole.exit_code == 0
    payload = _envelope(partial)
    # Same document a clean run emits -- nothing absent, nothing marked -- summing a
    # smaller set of files. Only the exit code and stderr say so.
    assert payload == _envelope(whole)
    assert [entry["source"]["name"] for entry in payload["results"]] == args
    assert set(payload) == {"schema_version", "results", "totals"}
    assert "Error: Path not found" in partial.stderr


def _two_file_args(tmp_path: Path) -> list[str]:
    first = tmp_path / "first.txt"
    first.write_text("hello world")
    second = tmp_path / "second.txt"
    second.write_text("goodbye world friend")
    return [str(first), str(second)]


def test_total_only_csv(tmp_path):
    args = _two_file_args(tmp_path)
    result = _invoke_cli(["--total-only", "--format", "csv", *args])
    assert result.exit_code == 0
    assert result.stdout.strip() == "TOTAL,6"


def test_total_only_csv_with_header(tmp_path):
    args = _two_file_args(tmp_path)
    result = _invoke_cli(["--header", "--total-only", "--format", "csv", *args])
    assert result.exit_code == 0
    assert result.stdout.strip() == "file,gpt-5\nTOTAL,6"


def test_file_csv_marks_each_approximate_model_column(tmp_path):
    args = _two_file_args(tmp_path)
    result = _invoke_cli(["--header", "--format", "csv", "-m", "gpt-6", *args])
    assert result.exit_code == 0
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "file,gpt-6_tokens,gpt-6_approximate"
    assert [line.split(",")[-2:] for line in lines[1:]] == [
        ["2", "true"],
        ["4", "true"],
    ]


def test_total_only_tsv(tmp_path):
    args = _two_file_args(tmp_path)
    result = _invoke_cli(["--total-only", "--format", "tsv", *args])
    assert result.exit_code == 0
    assert result.stdout.strip() == "TOTAL\t6"


def test_total_only_json(tmp_path):
    args = _two_file_args(tmp_path)
    result = _invoke_cli(["--total-only", "--format", "json", *args])
    assert result.exit_code == 0
    payload = _envelope(result)
    # The per-source rows go, the document does not change shape: both keys are
    # still there, so one reader handles a --total-only run and a full one.
    assert payload["results"] == []
    assert payload["totals"] == [
        {
            "model": "gpt-5",
            "tokens": 6,
            "approximate": False,
            "cost": None,
            "caveats": [],
            "retirement": None,
        }
    ]


def test_total_only_json_empties_results_for_a_text_run():
    result = _invoke_cli(
        ["--total-only", "--format", "json", "-m", "gpt-5", "--text", "hello world"]
    )

    assert result.exit_code == 0
    payload = _envelope(result)
    assert payload["results"] == []
    assert [total["model"] for total in payload["totals"]] == ["gpt-5"]
    assert payload["totals"][0]["tokens"] == 2


def test_total_only_json_empties_results_for_a_stdin_run():
    result = _invoke_cli(
        ["--total-only", "--format", "json", "-m", "gpt-5"], stdin="hello world"
    )

    assert result.exit_code == 0
    payload = _envelope(result)
    assert payload["results"] == []
    assert payload["totals"][0]["tokens"] == 2


def test_total_only_text(tmp_path, monkeypatch):
    monkeypatch.setattr("toko.cli.is_stdout_tty", lambda: True)
    args = _two_file_args(tmp_path)
    result = _invoke_cli(["--total-only", *args])
    assert result.exit_code == 0
    assert "TOTAL" in result.stdout
    assert "first.txt" not in result.stdout
    assert "second.txt" not in result.stdout


def test_total_only_does_not_add_a_model_column_to_text_input():
    plain = _invoke_cli(["--text", "hello world"])
    total_only = _invoke_cli(["--total-only", "--text", "hello world"])
    assert plain.exit_code == 0
    assert total_only.exit_code == 0
    assert plain.stdout.strip() == "2"
    assert total_only.stdout == plain.stdout


def test_total_only_leaves_multi_model_text_input_unchanged():
    args = ["-m", "gpt-5", "-m", "gpt-4.1", "--text", "hello world"]
    plain = _invoke_cli(args)
    total_only = _invoke_cli(["--total-only", *args])
    assert plain.exit_code == 0
    assert total_only.exit_code == 0
    assert plain.stdout.strip() == "gpt-5\t2\ngpt-4.1\t2"
    assert total_only.stdout == plain.stdout


def test_without_total_only_keeps_per_file_rows(tmp_path, monkeypatch):
    monkeypatch.setattr("toko.cli.is_stdout_tty", lambda: True)
    args = _two_file_args(tmp_path)
    result = _invoke_cli(args)
    assert result.exit_code == 0
    assert "first.txt" in result.stdout
    assert "second.txt" in result.stdout
    assert "TOTAL" in result.stdout


def test_without_total_only_keeps_per_file_rows_csv(tmp_path):
    args = _two_file_args(tmp_path)
    result = _invoke_cli(["--format", "csv", *args])
    assert result.exit_code == 0
    lines = [line for line in result.stdout.splitlines() if line]
    assert [line.split(",")[-1] for line in lines] == ["2", "4"]
    assert not any(line.startswith("TOTAL") for line in lines)


def test_invalid_format_is_a_usage_error_without_leaking_keys(tmp_path):
    sentinel = "sk-ant-FAKE-SENTINEL-XYZZY"
    config_dir = tmp_path / "toko"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        f'[toko.api_keys]\nanthropic = "{sentinel}"\n'
    )

    result = runner.invoke(
        app,
        ["--format", "jsonl", "--text", "hello"],
        env={"XDG_CONFIG_HOME": str(tmp_path)},
    )

    assert result.exit_code == 2
    combined = _normalize_cli_output(result.stdout + result.stderr)
    assert "Traceback" not in combined
    assert "api_keys" not in combined
    assert sentinel not in combined
    assert "'jsonl' is not one of" in combined


def test_invalid_default_format_in_config_is_a_clean_error(tmp_path):
    config_dir = tmp_path / "toko"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text('[toko]\ndefault_format = "jsonl"\n')

    result = runner.invoke(
        app, ["--text", "hello"], env={"XDG_CONFIG_HOME": str(tmp_path)}
    )

    assert result.exit_code == 1
    combined = _normalize_cli_output(result.stdout + result.stderr)
    assert "Traceback" not in combined
    assert "Invalid default_format 'jsonl'" in combined
    assert "text, json, csv, tsv" in combined


@pytest.mark.slow
def test_partial_success_missing_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert not os.environ.get("HF_TOKEN")

    text = "test-missing-hf-token-partial"
    with skip_if_rate_limited():
        result = _invoke_cli(
            ["--model", "gpt-5", "--model", "meta-llama/Llama-3.2-1B", "--text", text]
        )
    assert result.exit_code == 0
    expected = count_tokens(text, model="gpt-5")
    assert result.stdout.strip() == str(expected.count)
    assert "meta-llama/Llama-3.2-1B" not in result.stdout
    assert "Failed to count tokens for meta-llama/Llama-3.2-1B" in result.stderr
    assert "HF_TOKEN" in result.stderr


def _uneven_file_args(tmp_path: Path) -> list[str]:
    # Passed in an order that is neither path order nor count order, so each --sort value
    # lands on a different arrangement and no assertion below can pass by accident.
    (tmp_path / "a-small.txt").write_text("hello")
    (tmp_path / "b-mid.txt").write_text("one two three four five")
    (tmp_path / "c-big.txt").write_text("one two three four five six seven eight nine")
    return [str(tmp_path / name) for name in ("b-mid.txt", "c-big.txt", "a-small.txt")]


def _row_names(output: str, separator: str = "\t") -> list[str]:
    return [
        line.split(separator)[0].split("/")[-1]
        for line in output.splitlines()
        if line.strip()
    ]


def test_rows_follow_the_order_the_paths_were_given_by_default(tmp_path):
    result = _invoke_cli(_uneven_file_args(tmp_path))
    assert result.exit_code == 0
    assert _row_names(result.stdout) == ["b-mid.txt", "c-big.txt", "a-small.txt"]


def test_sort_input_matches_the_default(tmp_path):
    args = _uneven_file_args(tmp_path)
    assert _invoke_cli(["--sort", "input", *args]).stdout == _invoke_cli(args).stdout


def test_sort_path_reorders_paths_the_default_leaves_as_given(tmp_path):
    args = _uneven_file_args(tmp_path)
    sorted_run = _invoke_cli(["--sort", "path", *args])
    assert sorted_run.exit_code == 0
    assert _row_names(sorted_run.stdout) == ["a-small.txt", "b-mid.txt", "c-big.txt"]
    assert sorted_run.stdout != _invoke_cli(args).stdout


def test_sort_count_puts_the_largest_file_first(tmp_path):
    result = _invoke_cli(["--sort", "count", *_uneven_file_args(tmp_path)])
    assert result.exit_code == 0
    assert _row_names(result.stdout) == ["c-big.txt", "b-mid.txt", "a-small.txt"]


def test_sort_reaches_json_and_csv_too(tmp_path):
    args = _uneven_file_args(tmp_path)
    by_count = _invoke_cli(["--sort", "count", "--format", "json", *args])
    by_path = _invoke_cli(["--sort", "path", "--format", "csv", *args])
    assert [
        entry["source"]["name"].split("/")[-1]
        for entry in _envelope(by_count)["results"]
    ] == ["c-big.txt", "b-mid.txt", "a-small.txt"]
    assert _row_names(by_path.stdout, ",") == ["a-small.txt", "b-mid.txt", "c-big.txt"]


def test_sort_count_keeps_the_total_row_last(tmp_path, monkeypatch):
    monkeypatch.setattr("toko.cli.is_stdout_tty", lambda: True)
    result = _invoke_cli(["--sort", "count", *_uneven_file_args(tmp_path)])
    assert result.exit_code == 0
    # Row by row rather than through _normalize_cli_output, which folds the table into
    # one line; the borders come off so the assertion is only about the order.
    rows = [
        _BOX_DRAWING.sub(" ", _strip_ansi(line)).split()
        for line in result.stdout.splitlines()
    ]
    assert [row[0].split("/")[-1] for row in rows if row] == [
        "File",
        "c-big.txt",
        "b-mid.txt",
        "a-small.txt",
        "TOTAL",
    ]


def test_sort_is_accepted_and_ignored_for_text_input():
    plain = _invoke_cli(["--text", "hello world"])
    sorted_run = _invoke_cli(["--sort", "count", "--text", "hello world"])
    assert sorted_run.exit_code == 0
    assert sorted_run.stdout == plain.stdout


def test_unknown_sort_is_a_usage_error(tmp_path):
    result = _invoke_cli(["--sort", "size", *_uneven_file_args(tmp_path)])
    assert result.exit_code == 2
    combined = _normalize_cli_output(result.stdout + result.stderr)
    assert "'size' is not one of 'input', 'path', 'count'" in combined


def _write_tree(tmp_path: Path) -> Path:
    """Build a directory of files to count whose first file is by far the largest.

    Files are counted in sorted order, so the big one is submitted first and finishes
    last: a run that reported counts in completion order would put it elsewhere.
    """
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "aaa_big.txt").write_text("lorem ipsum dolor sit amet " * 20_000)
    for index in range(12):
        # Text the two OpenAI encodings disagree about, so a count filed under the wrong
        # model shows up as a changed number rather than passing unnoticed.
        (tree / f"file_{index:02d}.txt").write_text(
            f"héllo wörld — naïve café 日本語 {index}\n" * 3
        )
    return tree


def _csv_rows(output: str) -> list[list[str]]:
    return [line.split(",") for line in _strip_ansi(output).splitlines() if line]


def test_concurrent_counting_matches_a_sequential_run(tmp_path):
    assert DEFAULT_JOBS > 1, "the default has to be concurrent for this to compare"
    tree = _write_tree(tmp_path)
    # Two encodings, so a count landing under the wrong model would change the row.
    args = ["--header", "--format", "csv", "-m", "gpt-5", "-m", "gpt-4", str(tree)]

    # Concurrent first: counts are cached, so running the sequential pass first would
    # leave the concurrent one replaying that cache instead of counting anything.
    concurrent = _invoke_cli(args)
    sequential = _invoke_cli([*args, "--jobs", "1"])

    assert sequential.exit_code == 0
    assert concurrent.exit_code == 0
    assert concurrent.stdout == sequential.stdout

    rows = _csv_rows(concurrent.stdout)
    assert rows[0] == ["file", "gpt-5", "gpt-4"]
    assert rows[1][0] == str(tree / "aaa_big.txt")
    assert [row[0] for row in rows[1:]] == [
        str(path) for path in sorted(tree.iterdir())
    ]
    assert rows[2][1] != rows[2][2]


def test_concurrent_counting_reports_failures_like_a_sequential_run(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not os.environ.get("ANTHROPIC_API_KEY")
    tree = _write_tree(tmp_path)
    args = [
        "--header",
        "--format",
        "csv",
        "-m",
        "gpt-5",
        "-m",
        "claude-opus-4-5",
        str(tree),
    ]

    sequential = _invoke_cli([*args, "--jobs", "1"])
    concurrent = _invoke_cli(args)

    assert sequential.exit_code == 0
    assert concurrent.exit_code == 0
    assert concurrent.stdout == sequential.stdout
    # Compared line by line after sorting rather than verbatim: notices are printed by
    # whichever worker reaches them, so once two models have something to say their
    # order on stderr is whoever got there first. The content of every line is still
    # pinned; only the sequence between them is left free.
    assert sorted(concurrent.stderr.splitlines()) == sorted(
        sequential.stderr.splitlines()
    )
    assert (
        "Failed to count tokens for claude-opus-4-5 on 13 file(s)" in concurrent.stderr
    )
    assert "claude-opus-4-5" not in concurrent.stdout


def test_every_model_failing_still_exits_nonzero_when_concurrent(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not os.environ.get("ANTHROPIC_API_KEY")
    tree = _write_tree(tmp_path)

    result = _invoke_cli(["-m", "claude-opus-4-5", str(tree)])

    assert result.exit_code == 1
    assert "Error: All models failed for all files" in result.stderr


def test_a_warn_once_notice_is_still_printed_once_across_workers(tmp_path):
    """Every file re-reaches the notice, so the workers race for the one printing of it."""
    tree = _write_tree(tmp_path)

    result = _invoke_cli(["--format", "csv", "-m", "gpt-6-imaginary", str(tree)])

    assert result.exit_code == 0
    warnings = [
        line for line in result.stderr.splitlines() if line.startswith("Warning:")
    ]
    assert len(warnings) == 1
    assert "unknown OpenAI model 'gpt-6-imaginary'" in warnings[0]


def test_concurrent_counting_leaves_the_cache_intact(tmp_path):
    tree = _write_tree(tmp_path)

    result = _invoke_cli(
        ["--header", "--format", "csv", "-m", "gpt-5", "-m", "gpt-4", str(tree)]
    )

    assert result.exit_code == 0
    with sqlite3.connect(get_cache_db_path()) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    rows = _csv_rows(result.stdout)
    assert rows[0] == ["file", "gpt-5", "gpt-4"]
    assert len(rows) == 14
    for path, gpt_5, gpt_4 in rows[1:]:
        content = Path(path).read_text()
        # Every count the run reported is readable again under its own model, so no
        # write landed under another thread's key or was lost to a locked database.
        assert get_cached_count(content, "gpt-4") == int(gpt_4)
        assert get_cached_count(content, "gpt-5") == int(gpt_5)


def _tree_failing_the_first_file(tmp_path: Path, monkeypatch) -> Path:
    """Build a tree claude-opus-4-5 fails on the first file of but not the second.

    Nothing is faked: the API key really is absent, so every count against that model
    reaches the provider and fails -- except the one the cache answers before the
    provider runs, which is seeded for the second file only.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not os.environ.get("ANTHROPIC_API_KEY")
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a_first.txt").write_text("alpha beta gamma delta\n")
    later = tree / "b_later.txt"
    later.write_text("epsilon zeta eta theta\n")
    cache_count(later.read_text(), "claude-opus-4-5", 7)
    return tree


def test_columns_keep_model_order_when_a_model_fails_on_the_first_file(
    tmp_path, monkeypatch
):
    tree = _tree_failing_the_first_file(tmp_path, monkeypatch)

    result = _invoke_cli(
        [
            "--header",
            "--format",
            "csv",
            "-m",
            "claude-opus-4-5",
            "-m",
            "gpt-5",
            str(tree),
        ]
    )

    assert result.exit_code == 0
    rows = _csv_rows(result.stdout)
    # The order the models were named, not the order the counts arrived in: the first
    # file has no claude count at all, so a column order read off the results would
    # put gpt-5 first.
    assert rows[0] == ["file", "claude-opus-4-5", "gpt-5"]
    assert rows[1][1] == "N/A"
    assert rows[2][1] == "7"


def test_json_arrays_and_table_columns_agree_on_one_model_order(tmp_path, monkeypatch):
    tree = _tree_failing_the_first_file(tmp_path, monkeypatch)
    args = ["--header", "-m", "claude-opus-4-5", "-m", "gpt-5", str(tree)]

    table = _invoke_cli([*args, "--format", "csv"])
    document = _invoke_cli([*args, "--format", "json"])

    assert table.exit_code == 0
    assert document.exit_code == 0
    columns = _csv_rows(table.stdout)[0][1:]
    payload = json.loads(document.stdout)
    assert [count["model"] for count in payload["totals"]] == columns
    for source in payload["results"]:
        listed = [count["model"] for count in source["counts"]]
        # A source keeps only the models it could be counted for, but in the order the
        # columns use, so the two can be lined up.
        assert listed == [model for model in columns if model in listed]


def test_model_order_is_the_same_at_every_jobs_setting(tmp_path, monkeypatch):
    tree = _tree_failing_the_first_file(tmp_path, monkeypatch)
    args = [
        "--header",
        "--format",
        "json",
        "-m",
        "claude-opus-4-5",
        "-m",
        "gpt-5",
        "-m",
        "gpt-4",
        str(tree),
    ]

    outputs = [_invoke_cli([*args, "--jobs", str(jobs)]) for jobs in (MAX_JOBS, 8, 1)]

    assert [result.exit_code for result in outputs] == [0, 0, 0]
    orders = [
        [count["model"] for count in json.loads(result.stdout)["totals"]]
        for result in outputs
    ]
    assert orders == [["claude-opus-4-5", "gpt-5", "gpt-4"]] * 3
    assert outputs[0].stdout == outputs[1].stdout == outputs[2].stdout


def test_jobs_below_one_is_rejected(tmp_path):
    tree = _write_tree(tmp_path)

    result = _invoke_cli(["--jobs", "0", str(tree)])

    assert result.exit_code == 2
    assert "is not in the range" in _normalize_cli_output(result.stderr)


def test_jobs_above_the_cap_is_rejected(tmp_path):
    """An absurd -j is a usage error, not a thousand threads."""
    tree = _write_tree(tmp_path)

    result = _invoke_cli(["--jobs", str(MAX_JOBS + 1), str(tree)])

    assert result.exit_code == 2
    assert "is not in the range" in _normalize_cli_output(result.stderr)
