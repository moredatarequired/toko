# Toko

Toko is a token counting tool. It is built for use as a CLI, and available as a Python package.

## Highlights

- Accurate token counting for OpenAI models out of the box, with optional support for Anthropic, Google, xAI, Mistral, Llama, DeepSeek, and Qwen families.
- Reads inline text, stdin, files, directories (respects `.gitignore` automatically), and URLs.
- Compare multiple models in one run and add cost estimates powered by bundled `genai-prices` data.
- Emits `text`, `json`, `csv`, or `tsv` output. When stdout is piped, Toko automatically switches to TSV so you can chain tools like `cut` or `awk`.
- Caches counts in SQLite under your platform cache folder (e.g. `~/.cache/toko`) so repeated runs avoid redundant API calls.

## Install

Toko targets Python 3.14 and ships as a `uv` tool.

### Quick install

```sh
uv tool install toko
```

This places a `toko` executable on your `PATH`. Run `uv tool upgrade toko` to pick up new releases.

### Optional providers

Install extras when you need additional providers:

```sh
# HuggingFace tokenizers for Llama, DeepSeek, Qwen families
uv tool install 'toko[transformers]'

# Official Mistral tokenizer (mistral-common)
uv tool install 'toko[mistral]'

# Everything above in one go
uv tool install 'toko[all]'
```

If you are adding Toko to a project environment instead of the global toolchain, replace `uv tool install` with `uv add`.

### Source checkout (contributors)

```sh
git clone https://github.com/moredatarequired/toko
cd toko
just setup  # installs lefthook git hooks
```

## Quick start

Options in examples appear **before** any paths. `typer`/`click` treat everything after the first path argument as data input, so prefer `toko --total-only src` instead of `toko src --total-only`.

### Count inline text

```sh
toko --model gpt-5 --text "hello world"
```

```txt
2
```

If you omit `--model`, Toko falls back to your configured default. Fresh installs ship with `gpt-5`; override this in `config.toml` if your workflow needs a different model.

### Read a file

```sh
toko --model gpt-5 LICENSE
```

```txt
┏━━━━━━━━━┳━━━━━━━┓
┃ File    ┃ gpt-5 ┃
┡━━━━━━━━━╇━━━━━━━┩
│ LICENSE │   223 │
└─────────┴───────┘
```

Token counts will change if the file contents change.

### Stream from stdin

```sh
printf 'hello world' | toko --model gpt-5
```

```txt
2
```

When stdout is not a TTY (for example, when piping into another command) Toko emits TSV automatically and drops the header unless you pass `--header`.

### Compare models and estimate cost

```sh
toko --header --format tsv --model gpt-5-mini --model claude-opus-4-5 --text "The quick brown fox" --cost
```

```txt
model	tokens	cost
gpt-5-mini	4	$0.000005
claude-opus-4-5	11	$0.000055
```

Costs come from the bundled `genai-prices` feed. Models without pricing information display `N/A`.

### Work with directories, URLs, and filters

```sh
toko --exclude '**/__pycache__/*' src/
```

```txt
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ File                                 ┃  gpt-5 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ src/toko/__init__.py                 │     11 │
│ src/toko/cache.py                    │    758 │
│ src/toko/cli.py                      │  3,372 │
│ src/toko/config.py                   │    584 │
│ src/toko/cost.py                     │  1,325 │
│ src/toko/counter.py                  │  3,445 │
│ src/toko/data/__init__.py            │      8 │
│ src/toko/data/openrouter_models.json │    359 │
│ src/toko/file_reader.py              │  1,120 │
│ src/toko/formatters.py               │  2,226 │
│ src/toko/models.py                   │  4,211 │
│ src/toko/price_update.py             │    403 │
│ TOTAL                                │ 17,822 │
└──────────────────────────────────────┴────────┘
```

- Directories are processed recursively by default and honor `.gitignore`.
- Use `--no-recursive` to stay shallow and `--no-ignore` to include ignored files.

## Machine-readable output

Toko can emit structured output without post-processing.

```sh
toko --model gpt-5 --format json LICENSE
```

```json
{
  "LICENSE": {
    "gpt-5": 223
  }
}
```

```sh
toko --header --model gpt-5 --format csv --text "hello world"
```

```csv
model,tokens
gpt-5,2
```

Use `--format tsv` to force TSV even when running interactively.

## Know which models are available

```sh
toko --list-models | head -n 5
```

```txt
anthropic/claude-3-5-haiku-20241022
anthropic/claude-3-5-sonnet-20240620
anthropic/claude-3-5-sonnet-20241022
anthropic/claude-3-7-sonnet-20250219
anthropic/claude-3-haiku-20240307
```

## API keys and optional providers

Some providers require API credentials to access token counting endpoints:

- **Anthropic** – set `ANTHROPIC_API_KEY`
- **Google Gemini/Gemma** – set `GOOGLE_API_KEY`

Some providers use tokenizers available on Hugging Face; these may need authentication to download the appropriate tokenizer.

- **HuggingFace-hosted models (Llama, DeepSeek, Qwen)** – install `toko[transformers]` and ensure `huggingface-cli login` (or set `HF_TOKEN`) if the model needs authentication.
- **Mistral** – install `toko[mistral]`; no API key is required for offline tokenization.

Environment variables can be exported directly, stored in a `.env` file and loaded with `uv run --env-file`, or placed in the config file described below.

To mix providers, provide every required key. For example:

```sh
ANTHROPIC_API_KEY=sk-ant-... toko --model gpt-5 --model claude-sonnet-4-5 --text "Launch checklist" --cost
```

## Configuration

Toko reads configuration from `$XDG_CONFIG_HOME/toko/config.toml` (defaults to `~/.config/toko/config.toml`). A minimal example:

```toml
[toko]
default_model = "gpt-5"
default_format = "text"
respect_gitignore = true
auto_update_prices = false # fetch latest pricing when cached data is stale

[toko.exclude]
patterns = ["*.log", "*.tmp", "**/__pycache__/*"]

[toko.api_keys]
anthropic = "sk-ant-..."
openai = "sk-..."
```

Config values act as defaults; command-line flags always win.

## Caching and pricing data

- Counts are cached in `$XDG_CACHE_HOME/toko/token_cache.db`.
- Pricing data from `genai-prices` is stored alongside the package. When `auto_update_prices` is `true`, Toko silently refreshes the cache if data is older than a day. Fetch failures never abort your command.

## Development tasks

```sh
just lint          # Ruff check & format
just typecheck     # ty type checking
just test          # run "fast" tests
just check-all     # run the full set of checks
```

## License

MIT
