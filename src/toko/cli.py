"""CLI entry point for toko."""

import contextlib
import sys
from pathlib import Path
from typing import Annotated

import httpx
import typer
from genai_prices import UpdatePrices

from toko import __version__
from toko.cache import clear_cache as do_clear_cache
from toko.config import apply_api_keys, load_config
from toko.cost import estimate_cost
from toko.counter import count_tokens
from toko.file_reader import fetch_url, find_files, read_file
from toko.formatters import format_output, is_stdin_empty
from toko.models import list_models as get_model_list
from toko.price_update import update_prices_if_stale

app = typer.Typer(
    name="toko",
    help="A CLI-first token counting tool for LLMs",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"toko version {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Toko - Token counter for LLMs."""


@app.command()
def update_prices() -> None:
    """Update pricing data from genai-prices."""
    typer.echo("Fetching latest pricing data from genai-prices...")

    updater = UpdatePrices()
    result = updater.fetch()
    if result:
        typer.echo(
            f"✓ Successfully updated pricing data ({len(result.providers)} providers)"
        )
    else:
        typer.echo("✗ Failed to fetch pricing data", err=True)
        raise typer.Exit(1)


@app.command()
def clear_cache() -> None:
    """Clear the token count cache."""
    do_clear_cache()
    typer.echo("✓ Cache cleared")


@app.command()
def count(
    paths: Annotated[
        list[str] | None,
        typer.Argument(
            help="Files, directories, or URLs to count tokens for. If not provided, reads from stdin.",
        ),
    ] = None,
    model: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            "-m",
            help="Model to use for token counting (can be specified multiple times for comparison). Examples: gpt-4o, claude-sonnet-4, gemini-2.0-flash",
        ),
    ] = None,
    text: Annotated[
        str | None,
        typer.Option(
            "--text",
            "-t",
            help="Text string to count tokens for (alternative to files/stdin)",
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            "-e",
            help="Glob patterns to exclude (can be specified multiple times)",
        ),
    ] = None,
    no_ignore: Annotated[
        bool,
        typer.Option(
            "--no-ignore",
            help="Don't respect .gitignore files",
        ),
    ] = False,
    no_recursive: Annotated[
        bool,
        typer.Option(
            "--no-recursive",
            help="Don't recurse into directories",
        ),
    ] = False,
    total_only: Annotated[
        bool,
        typer.Option(
            "--total-only",
            help="Only show total count, not per-file breakdown",
        ),
    ] = False,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: text, json, csv, tsv",
        ),
    ] = "text",
    cost: Annotated[
        bool,
        typer.Option(
            "--cost",
            help="Show cost estimates using genai-prices",
        ),
    ] = False,
    list_models: Annotated[
        bool,
        typer.Option(
            "--list-models",
            help="List all supported models and exit",
        ),
    ] = False,
) -> None:
    """Count tokens in files, text, or stdin."""
    # Load config
    try:
        config = load_config()
        # Apply API keys from config if not already set in environment
        apply_api_keys(config)
    except ValueError as e:
        typer.echo(f"Error loading config: {e}", err=True)
        raise typer.Exit(1) from e

    # Auto-update prices if enabled and stale
    if config.auto_update_prices:
        # Silently update prices - don't fail command if update fails
        with contextlib.suppress(Exception):
            update_prices_if_stale()

    if list_models:
        typer.echo("Supported models:")
        models_by_provider = get_model_list()
        for provider, provider_models in models_by_provider.items():
            typer.echo(f"  {provider.capitalize()}: {', '.join(provider_models)}")
        raise typer.Exit

    # Use model from CLI or config default
    models = model or [config.default_model]

    # Use format from CLI or config default
    actual_format = output_format if output_format != "text" else config.default_format

    # Merge exclude patterns from CLI and config
    merged_exclude = list(config.exclude_patterns)
    if exclude:
        merged_exclude.extend(exclude)
    final_exclude = merged_exclude if merged_exclude else None

    # Determine input source
    input_text = None
    input_files: list[tuple[str, str]] = []  # (display_name, content) pairs

    if text:
        input_text = text
    elif paths:
        # Handle files, directories, and URLs
        for path_str in paths:
            # Check if URL
            if path_str.startswith(("http://", "https://")):
                try:
                    content = fetch_url(path_str)
                    input_files.append((path_str, content))
                except httpx.HTTPError as e:
                    typer.echo(f"Error fetching URL {path_str}: {e}", err=True)
                    raise typer.Exit(1) from e
                except UnicodeDecodeError as e:
                    typer.echo(
                        f"Error: URL content is not valid UTF-8: {path_str}",
                        err=True,
                    )
                    raise typer.Exit(1) from e
                except Exception as e:
                    typer.echo(f"Error fetching URL {path_str}: {e}", err=True)
                    raise typer.Exit(1) from e
            else:
                # File or directory
                try:
                    path = Path(path_str)
                    # Use CLI flags if set, otherwise use config defaults
                    should_respect_gitignore = (
                        config.respect_gitignore if not no_ignore else not no_ignore
                    )
                    files = find_files(
                        path,
                        recursive=not no_recursive,
                        respect_gitignore=should_respect_gitignore,
                        exclude_patterns=final_exclude,
                    )

                    for file_path in files:
                        try:
                            content = read_file(file_path)
                            # Use relative path if possible, otherwise absolute
                            try:
                                display_name = str(file_path.relative_to(Path.cwd()))
                            except ValueError:
                                display_name = str(file_path)
                            input_files.append((display_name, content))
                        except UnicodeDecodeError:
                            typer.echo(
                                f"Warning: Skipping binary file {file_path}",
                                err=True,
                            )
                        except Exception as e:
                            typer.echo(
                                f"Error reading {file_path}: {e}",
                                err=True,
                            )
                            raise typer.Exit(1) from e

                except (FileNotFoundError, ValueError) as e:
                    typer.echo(f"Error: {e}", err=True)
                    raise typer.Exit(1) from e

        if not input_files:
            typer.echo("Error: No files found matching criteria", err=True)
            raise typer.Exit(1)

    elif not is_stdin_empty():
        # Read from stdin
        input_text = sys.stdin.read()
    else:
        typer.echo(
            "Error: No input provided. Use --text, provide paths, or pipe to stdin.",
            err=True,
        )
        raise typer.Exit(1)

    # Count tokens for each model
    if input_text is not None:
        # Single text input (--text or stdin)
        results = {}
        for model_name in models:
            try:
                token_count = count_tokens(input_text, model=model_name)
                results[model_name] = token_count
            except ValueError as e:
                typer.echo(f"Error: {e}", err=True)
                raise typer.Exit(1) from e

        # Calculate costs if requested
        costs = None
        if cost:
            costs = {}
            for model_name, token_count in results.items():
                costs[model_name] = estimate_cost(token_count, model_name)

        # Format and output results
        output = format_output(
            results, output_format=actual_format, total_only=total_only, costs=costs
        )
        typer.echo(output)
    else:
        # Multiple files - count tokens for each file
        # For now, just combine all files and count total
        # TODO: Add per-file breakdown
        combined_text = "\n".join(content for _, content in input_files)
        results = {}
        for model_name in models:
            try:
                token_count = count_tokens(combined_text, model=model_name)
                results[model_name] = token_count
            except ValueError as e:
                typer.echo(f"Error: {e}", err=True)
                raise typer.Exit(1) from e

        # Calculate costs if requested
        costs = None
        if cost:
            costs = {}
            for model_name, token_count in results.items():
                costs[model_name] = estimate_cost(token_count, model_name)

        # Format and output results
        output = format_output(
            results, output_format=actual_format, total_only=total_only, costs=costs
        )
        typer.echo(output)


if __name__ == "__main__":
    app()
