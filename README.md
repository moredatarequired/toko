# Toko

Toko is a CLI-first token counting tool for modern LLMs. It is built for shell workflows, editor integrations, and CI pipelines where you need reliable counts without wiring up extra scripts.

## Highlights

- Accurate token counting for OpenAI and xAI models out of the box, with optional support for Anthropic, Google Gemini/Gemma, Mistral, Llama, DeepSeek, and Qwen families.
- Reads inline text, stdin, files, directories (respects `.gitignore` automatically), and HTTP URLs.
- Compare multiple models in one run and add cost estimates powered by bundled `genai-prices` data.
- Emits `text`, `json`, `csv`, or `tsv` output. When stdout is piped, Toko automatically switches to TSV so you can chain tools like `cut` or `awk`.
- Caches counts in SQLite so repeated runs avoid redundant API calls.

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
git clone https://github.com/hughwimberly/toko
cd toko
uv sync --all-groups
just setup  # installs lefthook git hooks
```

## Quick start

Options in examples appear **before** any paths. `typer`/`click` treat everything after the first path argument as data input, so prefer `toko --total-only src` instead of `toko src --total-only`.

### Count inline text

```sh
toko --model gpt-5 --text "hello world"
```

```txt
2 tokens
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
model	tokens
gpt-5	2
```

When stdout is not a TTY (for example, when piping into another command) Toko emits TSV automatically.

### Compare models and estimate cost

```sh
toko --model gpt-5 --model gpt-5-mini --text "The quick brown fox" --cost
```

```txt
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
┃ Model      ┃ Tokens ┃      Cost ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
│ gpt-5      │      4 │ $0.000005 │
│ gpt-5-mini │      4 │ $0.000001 │
└────────────┴────────┴───────────┘
```

Costs come from the bundled `genai-prices` feed. Models without pricing information display `N/A`.

### Work with directories, URLs, and filters

```sh
toko --model gpt-5 --exclude '**/__pycache__/*' src/
```

```txt
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ File                     ┃  gpt-5 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ src/toko/__init__.py     │     26 │
│ src/toko/cache.py        │    701 │
│ src/toko/cli.py          │  2,736 │
│ src/toko/config.py       │    641 │
│ src/toko/cost.py         │  1,454 │
│ src/toko/counter.py      │  1,791 │
│ src/toko/file_reader.py  │    947 │
│ src/toko/formatters.py   │  1,800 │
│ src/toko/models.py       │  2,620 │
│ src/toko/price_update.py │    403 │
│ TOTAL                    │ 13,119 │
└──────────────────────────┴────────┘
```

- Directories are processed recursively by default and honor `.gitignore`.
- Use `--no-recursive` to stay shallow and `--no-ignore` to include ignored files.
- URLs are fetched with `httpx`; invalid or non-UTF-8 responses fail with a clear error message.

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
toko --model gpt-5 --format csv --text "hello world"
```

```csv
model,tokens
gpt-5,2
```

Use `--format tsv` to force TSV even when running interactively.

## Know which models are available

```sh
toko --list-models
```

```txt
Supported models:
  Openai: ada, babbage, babbage-002, gpt-4, gpt-4.1, gpt-4o, gpt-5, o1, o3, ...
  Anthropic: claude-sonnet-4-5, claude-haiku-4-5, claude-3-7-sonnet-latest, ...
  Google: models/gemini-2.5-pro, models/gemini-2.5-flash, models/gemma-3-12b-it, ...
  Xai: grok-3, grok-3-mini, grok-4-fast-reasoning, ...
```

The full list includes every model shipped with the release plus pattern-based detection. Pass `--model <name>` to use any entry in the list (or a detectable future variant).

## API keys and optional providers

Some providers require API credentials:

- **Anthropic** – set `ANTHROPIC_API_KEY`
- **Google Gemini/Gemma** – set `GOOGLE_API_KEY`
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

- Counts are cached in `/tmp/toko/token_cache.db`. Delete the file to clear the cache.
- Pricing data from `genai-prices` is stored alongside the package. When `auto_update_prices` is `true`, Toko silently refreshes the cache if data is older than a day. Fetch failures never abort your command.

## Development tasks

```sh
just lint          # Ruff check & format
just typecheck     # ty type checking
just test          # pytest
just check-all     # run the full pre-commit hook chain
```

## License

MIT
