# Toko

A CLI-first token counting tool for LLMs that actually meets your needs.

## Features

- **Multi-model support**: Count tokens for OpenAI (via tiktoken), Anthropic, Google, and xAI models
- **Flexible input**: stdin, text strings, files, directories, or URLs
- **Smart defaults**: Recursive directory scanning with .gitignore respect (like ripgrep)
- **Model comparison**: Compare token counts across multiple models simultaneously
- **Cost estimation**: Built-in cost estimates via genai-prices
- **Multiple output formats**: Human-friendly or machine-parseable (text, JSON, CSV, TSV)

## Installation

```sh
uv tool install toko
```

Or for development:

```sh
git clone https://github.com/yourusername/toko
cd toko
uv sync --all-groups
just setup
```

## Usage

```sh
# Count tokens from stdin
echo "hello world" | toko

# Count tokens in a file
toko myfile.txt

# Count tokens in a directory (recursive by default, respects .gitignore)
toko src/

# Count tokens from a text string
toko --text "hello world"

# Compare across multiple models
toko --model gpt-4o --model claude-sonnet-4 src/

# Show cost estimates
toko --cost src/

# Get just the total
toko --total-only src/

# Machine-readable output
toko --format json src/

# Exclude patterns
toko --exclude "*.test.js" --exclude "*.md" src/

# Count tokens from a URL
toko https://example.com/file.txt

# List all supported models
toko --list-models
```

## Configuration

Toko looks for configuration at `$XDG_CONFIG_HOME/toko/config.toml` (usually `~/.config/toko/config.toml`).

Example config:

```toml
[toko]
default_model = "gpt-4o"
respect_gitignore = true
default_format = "text"

[toko.exclude]
patterns = ["*.log", "*.tmp", "node_modules/*"]

[toko.api_keys]  # optional
anthropic = "sk-..."
openai = "sk-..."
```

## Development

This project uses modern Python tooling:

- **uv**: Package management
- **just**: Task runner
- **Ruff**: Linting and formatting
- **ty**: Type checking
- **Lefthook**: Git hooks
- **pytest**: Testing

Common commands:

```sh
just setup          # Install git hooks
just lint          # Lint and format code
just test          # Run tests
just test-coverage # Run tests with coverage
just typecheck     # Type check the code
just check-all     # Run all pre-commit checks
```

## License

MIT
