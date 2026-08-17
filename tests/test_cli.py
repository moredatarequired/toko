"""Tests for the CLI."""

import ast
import json
import os
import re
import sqlite3
from pathlib import Path

import httpx
import pytest
import respx
from genai_prices.data_snapshot import set_custom_snapshot
from typer.testing import CliRunner

from tests.hf_hub import skip_if_rate_limited
from toko.cache import get_cache_db_path, get_cached_count
from toko.cli import DEFAULT_JOBS, MAX_JOBS, app
from toko.cost import format_cost, format_cost_value
from toko.counter import ANTHROPIC_COUNT_URL, GOOGLE_COUNT_URL_BASE, count_tokens
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


def test_a_retired_openai_engine_is_refused_without_the_flag(tmp_path):
    """RETIRED_OPENAI_MODELS was silent before; naming one is now an error."""
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world")

    result = _invoke_cli(["--format", "csv", "-m", "text-davinci-003", str(sample)])

    assert result.exit_code == 1
    assert "model 'text-davinci-003' is retired (date unknown)." in result.stderr
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


def test_an_opted_in_retired_model_carries_its_retirement_into_json(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "")

    result = _invoke_cli(
        ["--include-retired", "--format", "json", "-m", "grok-3", "--text", "hi"],
        {"XAI_API_KEY": ""},
    )

    if result.exit_code != 0:
        pytest.skip("counting grok-3 needs the xAI API or the Grok-1 tokenizer")
    assert _only_count(result)["retirement"] == {
        "model": "grok-3",
        "date": "2026-05-15",
        "redirects_to": "grok-4.3",
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

    results[].counts and totals hold one count object per model in the same shape, so a
    reader zips them by index, and two orders in one document misattribute silently.
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
