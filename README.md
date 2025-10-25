# Toko

A CLI-first token counting tool for LLMs that actually meets your needs.

## Features

### ✅ Implemented

- **52 models across 3 providers**: 30 OpenAI (tiktoken), 8 Anthropic, 14 Google models
- **Text and stdin input**: Count tokens from strings or piped input
- **Model comparison**: Compare token counts across multiple models simultaneously
- **Multiple output formats**: text, JSON, CSV, TSV
- **Environment-based API keys**: Load from `.env` file or environment variables

### 🚧 Roadmap

- **File/directory support**: Count tokens in files and directories (recursive by default)
- **Smart defaults**: Respect .gitignore files (like ripgrep)
- **Exclude patterns**: Filter files with glob patterns
- **URL support**: Count tokens from remote URLs
- **Cost estimation**: Built-in cost estimates via genai-prices
- **Config file support**: `~/.config/toko/config.toml` for defaults

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

### Currently Working

```sh
# Count tokens from a text string
toko count --text "hello world"

# Count tokens from stdin
echo "hello world" | toko count

# Compare across multiple models
toko count --model gpt-4o --model claude-3-5-haiku-20241022 --text "hello"

# Machine-readable output
toko count --format json --text "hello world"

# List all supported models
toko count --list-models
```

### With API Keys (Anthropic, Google)

Set up your `.env` file:

```bash
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

Then use with `uv run --env-file .env`:

```sh
uv run --env-file .env toko count \
  --model gpt-4o \
  --model claude-3-5-haiku-20241022 \
  --model gemini-2.5-flash \
  --text "The quick brown fox"
```

### Coming Soon

```sh
# Count tokens in files and directories
toko count myfile.txt
toko count src/

# Exclude patterns
toko count --exclude "*.test.js" --exclude "*.md" src/

# Count tokens from a URL
toko count https://example.com/file.txt

# Show cost estimates
toko count --cost src/
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

[toko.api_keys] # optional
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
