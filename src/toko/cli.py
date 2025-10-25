"""CLI entry point for toko."""

from typing import Annotated

import typer

from toko import __version__

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
    if list_models:
        typer.echo("Supported models:")
        typer.echo("  OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo")
        typer.echo("  Anthropic: claude-opus-4, claude-sonnet-4, claude-haiku-4")
        typer.echo("  Google: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash")
        typer.echo("  xAI: grok-2, grok-2-mini")
        raise typer.Exit

    # Default model
    models = model or ["gpt-4o"]

    typer.echo(f"Counting tokens with model(s): {', '.join(models)}")
    typer.echo(f"Paths: {paths}")
    typer.echo(f"Text: {text}")
    typer.echo(f"Exclude: {exclude}")
    typer.echo(f"No ignore: {no_ignore}")
    typer.echo(f"No recursive: {no_recursive}")
    typer.echo(f"Total only: {total_only}")
    typer.echo(f"Format: {output_format}")
    typer.echo(f"Show cost: {cost}")
    typer.echo("\nTODO: Implement token counting logic")


if __name__ == "__main__":
    app()
