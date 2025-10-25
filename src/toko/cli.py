"""CLI entry point for toko."""

import sys
from pathlib import Path
from typing import Annotated

import typer

from toko import __version__
from toko.counter import count_tokens
from toko.file_reader import find_files, read_file
from toko.formatters import format_output, is_stdin_empty
from toko.models import list_models as get_model_list

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
    cost: Annotated[  # noqa: ARG001
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
    if list_models:
        typer.echo("Supported models:")
        models_by_provider = get_model_list()
        for provider, provider_models in models_by_provider.items():
            typer.echo(f"  {provider.capitalize()}: {', '.join(provider_models)}")
        raise typer.Exit

    # Default model
    models = model or ["gpt-4o"]

    # Determine input source
    input_text = None
    input_files: list[tuple[str, str]] = []  # (display_name, content) pairs

    if text:
        input_text = text
    elif paths:
        # Handle files and directories
        for path_str in paths:
            # TODO: Check if URL and handle accordingly
            if path_str.startswith(("http://", "https://")):
                typer.echo(
                    f"Error: URL support not yet implemented: {path_str}", err=True
                )
                raise typer.Exit(1)

            try:
                path = Path(path_str)
                files = find_files(
                    path,
                    recursive=not no_recursive,
                    respect_gitignore=not no_ignore,
                    exclude_patterns=exclude,
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

        # Format and output results
        output = format_output(
            results, output_format=output_format, total_only=total_only
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

        # Format and output results
        output = format_output(
            results, output_format=output_format, total_only=total_only
        )
        typer.echo(output)


if __name__ == "__main__":
    app()
