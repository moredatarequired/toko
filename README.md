# Toko

A CLI-first token counting tool for LLMs that actually meets your needs.

## Features

### ✅ Implemented

- **60 models across 4 providers**: 30 OpenAI (tiktoken), 8 Anthropic, 14 Google, 8 xAI models
- **Multiple input methods**: Text strings, stdin, files, directories (recursive), URLs
- **Smart defaults**: Respects .gitignore files (like ripgrep)
- **Exclude patterns**: Filter files with glob patterns
- **Model comparison**: Compare token counts across multiple models simultaneously
- **Multiple output formats**: text (with rich tables), JSON, CSV, TSV
- **Cost estimation**: Built-in cost estimates via genai-prices
- **Config file support**: `~/.config/toko/config.toml` for defaults
- **Price updates**: Manual price data updates via `toko update-prices`
- **Environment-based API keys**: Load from `.env` file, environment variables, or config file

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

### Basic Usage

```sh
# Count tokens from a text string
toko count --text "hello world"

# Count tokens from stdin
echo "hello world" | toko count

# Count tokens from a file
toko count myfile.txt

# Count tokens from a directory (recursive, respects .gitignore)
toko count src/

# Count tokens from a URL
toko count https://raw.githubusercontent.com/user/repo/main/README.md

# List all supported models
toko count --list-models
```

### Model Comparison

```sh
# Compare across multiple models
toko count --model gpt-4o --model claude-3-5-haiku-20241022 --text "hello"

# Output (with beautiful rich tables):
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Model                     ┃ Tokens ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ gpt-4o                    │      1 │
│ claude-3-5-haiku-20241022 │      8 │
└───────────────────────────┴────────┘
```

### Cost Estimation

```sh
# Show cost estimates
toko count --model gpt-4o --model claude-3-5-haiku-20241022 --text "hello" --cost

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
┃ Model                     ┃ Tokens ┃      Cost ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
│ gpt-4o                    │      1 │ $0.000002 │
│ claude-3-5-haiku-20241022 │      8 │ $0.000007 │
└───────────────────────────┴────────┴───────────┘

# Update pricing data
toko update-prices
```

### File Operations

```sh
# Exclude patterns
toko count --exclude "*.test.js" --exclude "*.md" src/

# Don't respect .gitignore
toko count --no-ignore src/

# Don't recurse into subdirectories
toko count --no-recursive src/

# Machine-readable output
toko count --format json src/
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

Or add API keys to your config file (see Configuration below).

## Configuration

Toko looks for configuration at `$XDG_CONFIG_HOME/toko/config.toml` (usually `~/.config/toko/config.toml`).

Example config:

```toml
[toko]
default_model = "gpt-4o"
respect_gitignore = true
default_format = "text"
auto_update_prices = false # Set to true to auto-update pricing data (if > 1 day old)

[toko.exclude]
patterns = ["*.log", "*.tmp", "node_modules/*"]

[toko.api_keys] # optional
anthropic = "sk-..."
openai = "sk-..."
```

### Auto-Update Prices

By default, toko does NOT automatically update pricing data to avoid unnecessary network calls. However, you can enable automatic updates:

**Via config file:**

```toml
[toko]
auto_update_prices = true
```

**Via environment variable:**

```bash
export TOKO_AUTO_UPDATE_PRICES=true
```

When enabled, toko will automatically fetch fresh pricing data from genai-prices if the local data is more than 1 day old. This happens transparently in the background and won't fail your commands if the update fails.

### Token Count Caching

Toko automatically caches token counts in a local SQLite database to avoid redundant API calls and speed up repeated queries. The cache is keyed by message hash, so identical text will return cached results instantly.

**Cache location:** `/tmp/toko/token_cache.db`

**Clear the cache:**

```bash
toko clear-cache
```

The cache stores a JSON object mapping model names to token counts for each unique message, allowing efficient multi-model comparisons on the same text.

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
