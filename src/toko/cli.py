"""CLI entry point for toko."""

import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from typer.core import TyperGroup

from toko import __version__
from toko.cache import clear_cache as do_clear_cache
from toko.config import Config, apply_api_keys, load_config
from toko.cost import estimate_cost
from toko.counter import count_tokens
from toko.file_reader import fetch_url, find_files, read_file
from toko.formatters import format_file_table, format_output, is_stdin_empty
from toko.models import list_models as get_model_list
from toko.output_format import OutputFormat
from toko.price_update import (
    apply_cached_prices,
    clear_price_cache,
    refresh_prices,
    update_prices_if_stale,
)

if TYPE_CHECKING:
    import click

_SUBCOMMAND_META_KEY = "toko.subcommand"


def is_stdout_tty() -> bool:
    """Check if stdout is connected to a TTY (not piped)."""
    return sys.stdout.isatty()


def is_stderr_tty() -> bool:
    """Check if stderr is connected to a TTY."""
    return sys.stderr.isatty()


class TokoGroup(TyperGroup):
    """Keep subcommands reachable alongside the default command's variadic PATHS.

    Click fills the group's own arguments before it looks for a subcommand, so PATHS
    would otherwise swallow names like ``update-prices`` and treat them as files.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        is_subcommand = bool(args) and args[0] in self.commands
        ctx.meta[_SUBCOMMAND_META_KEY] = is_subcommand
        # click.Group defaults allow_interspersed_args to False so that options after
        # a subcommand name reach the subcommand. The default command needs the
        # opposite (`toko src --total-only` must not read the flag as a path), so pick
        # per invocation rather than setting it globally. The parser is built from ctx
        # inside super().parse_args, so this lands before it is read.
        ctx.allow_interspersed_args = not is_subcommand
        return super().parse_args(ctx, args)

    def get_params(self, ctx: click.Context) -> list[click.Parameter]:
        params = super().get_params(ctx)
        if ctx.meta.get(_SUBCOMMAND_META_KEY):
            return [param for param in params if param.name != "paths"]
        return params


app = typer.Typer(
    name="toko",
    help="A CLI-first token counting tool for LLMs",
    no_args_is_help=False,
    invoke_without_command=True,
    cls=TokoGroup,
    # Frame locals hold API keys; keep them out of any traceback Typer renders.
    pretty_exceptions_show_locals=False,
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"toko version {__version__}")
        raise typer.Exit


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    _version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
    paths: Annotated[
        list[str] | None,
        typer.Argument(help="Files, directories, or URLs to count tokens for"),
    ] = None,
    model: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            "-m",
            help="Model to use for token counting (can be specified multiple times)",
        ),
    ] = None,
    text: Annotated[
        str | None, typer.Option("--text", "-t", help="Text string to count tokens for")
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", "-e", help="Glob patterns to exclude"),
    ] = None,
    no_ignore: Annotated[
        bool, typer.Option("--no-ignore", help="Don't respect .gitignore files")
    ] = False,
    no_recursive: Annotated[
        bool, typer.Option("--no-recursive", help="Don't recurse into directories")
    ] = False,
    total_only: Annotated[
        bool,
        typer.Option(
            "--total-only", help="Only show total count, not per-file breakdown"
        ),
    ] = False,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", "-f", help="Output format")
    ] = OutputFormat.TEXT,
    cost: Annotated[bool, typer.Option("--cost", help="Show cost estimates")] = False,
    header: Annotated[
        bool | None,
        typer.Option(
            "--header/--no-header", help="Include header row in tabular outputs"
        ),
    ] = None,
    list_models: Annotated[
        bool, typer.Option("--list-models", help="List all supported models and exit")
    ] = False,
) -> None:
    """Toko - Token counter for LLMs."""
    # If a subcommand was invoked, don't run default behavior
    if ctx.invoked_subcommand is not None:
        return

    # Otherwise, run the count logic as default
    _do_count(
        paths,
        model,
        text,
        exclude,
        no_ignore,
        no_recursive,
        total_only,
        output_format,
        cost,
        header,
        list_models,
    )


@app.command()
def update_prices() -> None:
    """Update pricing data from genai-prices."""
    typer.echo("Fetching latest pricing data from genai-prices...")

    try:
        provider_count = refresh_prices()
    except Exception as e:
        typer.echo(f"✗ Failed to fetch pricing data: {e}", err=True)
        raise typer.Exit(1) from e

    typer.echo(f"✓ Successfully updated pricing data ({provider_count} providers)")


@app.command()
def clear_cache() -> None:
    """Clear the token count cache and the cached pricing data."""
    do_clear_cache()
    clear_price_cache()
    typer.echo("✓ Cache cleared")


@dataclass
class InputSelection:
    text: str | None
    files: list[tuple[str, str]]
    had_failures: bool = False


def _load_runtime_config() -> Config:
    try:
        config = load_config()
        apply_api_keys(config)
    except ValueError as e:
        typer.echo(f"Error loading config: {e}", err=True)
        raise typer.Exit(1) from e

    return config


def _prepare_prices(config: Config) -> None:
    # Prices previously fetched by `toko update-prices` are installed on every run;
    # only the download itself is opt-in, otherwise the manual command would have no
    # effect for anyone who has not enabled auto-updates.
    apply_cached_prices()
    if not config.auto_update_prices:
        return
    with contextlib.suppress(Exception):
        update_prices_if_stale()


def _resolve_models(config: Config, cli_models: list[str] | None) -> list[str]:
    return cli_models or [config.default_model]


def _resolve_output_format(config: Config, requested: OutputFormat) -> OutputFormat:
    if requested == OutputFormat.TEXT:
        return config.default_format if is_stdout_tty() else OutputFormat.TSV
    return requested


def _merge_excludes(config: Config, cli_excludes: list[str] | None) -> list[str] | None:
    patterns = list(config.exclude_patterns)
    if cli_excludes:
        patterns.extend(cli_excludes)
    return patterns or None


def _collect_inputs(
    paths: list[str] | None,
    text: str | None,
    config: Config,
    *,
    no_ignore: bool,
    no_recursive: bool,
    exclude_patterns: list[str] | None,
) -> InputSelection:
    if text:
        return InputSelection(text=text, files=[])

    if paths:
        files, had_failures = _collect_files_from_paths(
            paths,
            config,
            no_ignore=no_ignore,
            no_recursive=no_recursive,
            exclude_patterns=exclude_patterns,
        )
        if not files:
            # A per-path error was already reported; don't pile a vague one on top.
            if not had_failures:
                typer.echo("Error: No files found matching criteria", err=True)
            raise typer.Exit(1)
        return InputSelection(text=None, files=files, had_failures=had_failures)

    if not is_stdin_empty():
        stdin_text = sys.stdin.read()
        return InputSelection(text=stdin_text, files=[])

    typer.echo(
        "Error: No input provided. Use --text, provide paths, or pipe to stdin.",
        err=True,
    )
    raise typer.Exit(1)


def _collect_files_from_paths(
    paths: list[str],
    config: Config,
    *,
    no_ignore: bool,
    no_recursive: bool,
    exclude_patterns: list[str] | None,
) -> tuple[list[tuple[str, str]], bool]:
    collected: list[tuple[str, str]] = []
    should_respect_gitignore = config.respect_gitignore if not no_ignore else False
    had_failures = False

    for path_str in paths:
        if path_str.startswith(("http://", "https://")):
            ok = _collect_from_url(path_str, collected)
        else:
            ok = _collect_from_filesystem(
                Path(path_str),
                collected,
                recursive=not no_recursive,
                respect_gitignore=should_respect_gitignore,
                exclude_patterns=exclude_patterns,
            )
        had_failures = had_failures or not ok

    return collected, had_failures


def _collect_from_url(path_str: str, collected: list[tuple[str, str]]) -> bool:
    try:
        content = fetch_url(path_str)
    except UnicodeDecodeError:
        typer.echo(f"Error: URL content is not valid UTF-8: {path_str}", err=True)
        return False
    except Exception as e:
        typer.echo(f"Error fetching URL {path_str}: {e}", err=True)
        return False

    collected.append((path_str, content))
    return True


def _collect_from_filesystem(
    path: Path,
    collected: list[tuple[str, str]],
    *,
    recursive: bool,
    respect_gitignore: bool,
    exclude_patterns: list[str] | None,
) -> bool:
    try:
        files = find_files(
            path,
            recursive=recursive,
            respect_gitignore=respect_gitignore,
            exclude_patterns=exclude_patterns,
        )
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        return False

    ok = True
    for file_path in files:
        try:
            content = read_file(file_path)
        except UnicodeDecodeError:
            typer.echo(f"Warning: Skipping binary file {file_path}", err=True)
            continue
        except Exception as e:
            typer.echo(f"Error reading {file_path}: {e}", err=True)
            ok = False
            continue

        try:
            display_name = str(file_path.relative_to(Path.cwd()))
        except ValueError:
            display_name = str(file_path)

        collected.append((display_name, content))

    return ok


def _handle_text_input(
    models: list[str],
    text: str,
    *,
    output_format: OutputFormat,
    include_costs: bool,
    include_header: bool,
) -> None:
    results: dict[str, int] = {}

    for model_name in models:
        try:
            results[model_name] = count_tokens(text, model=model_name)
        except ValueError as e:
            typer.echo(
                f"Warning: Failed to count tokens for {model_name}: {e}", err=True
            )

    if not results:
        typer.echo("Error: All models failed to count tokens", err=True)
        raise typer.Exit(1)

    costs = _calculate_costs(results, include_costs)

    adjusted_format = output_format
    if (
        output_format == OutputFormat.TSV
        and not include_costs
        and not include_header
        and len(results) == 1
    ):
        adjusted_format = OutputFormat.TEXT

    output = format_output(
        results,
        output_format=adjusted_format,
        costs=costs,
        include_header=include_header,
    )
    typer.echo(output)


def _collect_file_counts(
    models: list[str], files: list[tuple[str, str]]
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, str]], dict[str, str]]:
    file_results: dict[str, dict[str, int]] = {}
    file_errors: dict[str, dict[str, str]] = {}
    model_errors: dict[str, str] = {}

    for display_name, content in files:
        results = file_results.setdefault(display_name, {})
        errors = file_errors.setdefault(display_name, {})

        for model_name in models:
            try:
                results[model_name] = count_tokens(content, model=model_name)
            except ValueError as exc:
                error_msg = str(exc)
                errors[model_name] = error_msg
                model_errors.setdefault(model_name, error_msg)

    return file_results, file_errors, model_errors


def _emit_model_error_summary(
    model_errors: dict[str, str], file_errors: dict[str, dict[str, str]]
) -> None:
    for model_name, error_msg in model_errors.items():
        failed_files = [
            filename for filename, errors in file_errors.items() if model_name in errors
        ]
        if not failed_files:
            continue
        typer.echo(
            f"Warning: Failed to count tokens for {model_name} on {len(failed_files)} file(s)"
            f" (including {failed_files[0]}): {error_msg}",
            err=True,
        )


def _handle_file_inputs(
    models: list[str],
    files: list[tuple[str, str]],
    *,
    output_format: OutputFormat,
    total_only: bool,
    include_costs: bool,
    include_header: bool,
) -> None:
    file_results, file_errors, model_errors = _collect_file_counts(models, files)

    if model_errors:
        _emit_model_error_summary(model_errors, file_errors)

    has_results = any(file_results[file] for file in file_results)
    if not has_results:
        typer.echo("Error: All models failed for all files", err=True)
        raise typer.Exit(1)

    file_costs = _calculate_file_costs(file_results) if include_costs else None

    output = format_file_table(
        file_results,
        output_format=output_format,
        total_only=total_only,
        costs=file_costs,
        include_header=include_header,
    )
    typer.echo(output)


def _calculate_costs(
    counts: dict[str, int], include_costs: bool
) -> dict[str, float | None] | None:
    if not include_costs:
        return None
    return {
        model_name: estimate_cost(tokens, model_name)
        for model_name, tokens in counts.items()
    }


def _calculate_file_costs(
    file_counts: dict[str, dict[str, int]],
) -> dict[str, dict[str, float | None]]:
    costs: dict[str, dict[str, float | None]] = {}
    for filename, model_counts in file_counts.items():
        costs[filename] = {}
        for model_name, token_count in model_counts.items():
            costs[filename][model_name] = estimate_cost(token_count, model_name)
    return costs


def _format_model_name(provider: str, model: str) -> str:
    base = model
    if base.startswith("models/"):
        base = base.split("/", 1)[1]
    if "/" in base:
        return base
    return f"{provider}/{base}"


def _collect_supported_models() -> list[str]:
    models_by_provider = get_model_list()
    names: set[str] = set()
    for provider, provider_models in models_by_provider.items():
        for model in provider_models:
            names.add(_format_model_name(provider, model))
    return sorted(names, key=str.lower)


def _show_model_list() -> None:
    models = _collect_supported_models()
    typer.echo("\n".join(models))
    raise typer.Exit


def _do_count(
    paths: list[str] | None,
    model: list[str] | None,
    text: str | None,
    exclude: list[str] | None,
    no_ignore: bool,
    no_recursive: bool,
    total_only: bool,
    output_format: OutputFormat,
    cost: bool,
    header: bool | None,
    list_models: bool,
) -> None:
    config = _load_runtime_config()
    _prepare_prices(config)
    if list_models:
        _show_model_list()
    models = _resolve_models(config, model)
    actual_format = _resolve_output_format(config, output_format)
    merged_exclude = _merge_excludes(config, exclude)
    include_header = header if header is not None else is_stdout_tty()
    inputs = _collect_inputs(
        paths,
        text,
        config,
        no_ignore=no_ignore,
        no_recursive=no_recursive,
        exclude_patterns=merged_exclude,
    )
    if inputs.text is not None:
        _handle_text_input(
            models,
            inputs.text,
            output_format=actual_format,
            include_costs=cost,
            include_header=include_header,
        )
        return
    _handle_file_inputs(
        models,
        inputs.files,
        output_format=actual_format,
        total_only=total_only,
        include_costs=cost,
        include_header=include_header,
    )
    # Results for the readable inputs are already printed; signal the bad ones.
    if inputs.had_failures:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
