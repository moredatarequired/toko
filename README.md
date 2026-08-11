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

Options may appear before or after paths, so `toko --total-only src` and `toko src --total-only` are equivalent.

A first argument that names a subcommand (`update-prices`, `clear-cache`) runs that subcommand rather than counting a file of the same name. Only the first position is treated this way, so `toko README.md update-prices` counts both files, and `toko -- update-prices` counts a file literally named `update-prices`.

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
gpt-5-mini	4	$0.000001
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

### What a piped run emits without `--format`

Since a non-TTY stdout already defaults to headerless TSV, a piped or captured run needs
no `--format` at all. A single-model `--text` or stdin run collapses one step further,
to a bare number, which is what makes the usual scripting shape work:

```sh
n=$(toko -m gpt-5 --text "hello world")   # 2
```

Two exceptions are worth knowing before you parse the output:

- The collapse to a bare number is for `--text` and stdin only. Give `toko` a path and
  you get a `file<TAB>tokens` row per file, because the filename has to go somewhere.
- An **approximate** count never collapses. It keeps its full row so the marker travels
  with the number, rather than being stranded on stderr where the process on the other
  end of the pipe cannot see it. `n=$(toko -m gpt-6 --text "hello world")` yields
  `gpt-6<TAB>2<TAB>true`, not `2`. Read the second field, or pass `--format json` and
  read `tokens`, if a run of yours can hit an unreleased OpenAI name or an xAI model
  without `XAI_API_KEY` — the two cases that produce an approximate count.

## Library usage

`count_tokens` is the whole public API, and it returns a `TokenCount` describing how the
number was reached:

```python
from toko import count_tokens

result = count_tokens("hello world", model="gpt-5")
print(result.count)  # 2
```

| Field         | Type            | Meaning                                                                               |
| ------------- | --------------- | ------------------------------------------------------------------------------------- |
| `count`       | `int`           | The token count.                                                                      |
| `model`       | `str`           | The canonical model the request resolved to, which need not be the string you passed. |
| `provider`    | `str`           | The provider that produced the count.                                                 |
| `approximate` | `bool`          | True when the count came from a stand-in tokenizer rather than the model's own.       |
| `caveat`      | `str \| None`   | Why the count is approximate, when it is.                                             |
| `cost`        | `float \| None` | Estimated cost in USD, when pricing data covers the model.                            |

An unreachable tokenizer is usually an error, not a fallback: `count_tokens` raises
`ValueError` when `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY` is missing, and when no
provider can be detected from the name at all. Exactly two paths return an approximate
count instead, so check `approximate` on those before treating a count as exact:

- An OpenAI name Toko does not know yet — one that still reads as an OpenAI model, such
  as `gpt-6` — is counted with `o200k_base`, the encoding every OpenAI model since
  `gpt-4o` uses.
- An xAI model is counted with the Grok-1 Hugging Face tokenizer when `XAI_API_KEY` is
  unset or the xAI endpoint fails. That stand-in needs `toko[transformers]`; without it
  the call raises instead.

```python
result = count_tokens("hello world", model="grok-4")
if result.approximate:
    print(f"estimate only: {result.caveat}")
```

`caveat` is a human-readable explanation to show a person and is explicitly **not**
machine-parseable — its wording, and for an aggregate its internal punctuation, may change
at any time, so branch on the `approximate` boolean rather than parsing `caveat`.

`TokenCount` is deliberately not int-like: it does not implement `__int__`/`__index__`
or arithmetic, and it compares equal only to another `TokenCount` with the same fields,
so use `result.count` wherever you need the number. Most misuses fail loudly — `+`,
`int()`, `sum()`, `json.dumps`, and `%d` all raise — but three are quiet:

- `result == 2` is `False` rather than an error.
- `bool(result)` is always `True`, so a zero-token count is truthy.
- `f"{result}"` renders the dataclass repr; use `f"{result.count}"`.

## Know which models are available

```sh
toko --list-models | head -n 5
```

```txt
anthropic/claude-fable-5
anthropic/claude-haiku-4-5-20251001
anthropic/claude-opus-4-5-20251101
anthropic/claude-opus-4-6
anthropic/claude-opus-4-7
```

Retired models are hidden from that listing: both the dead OpenAI engines tiktoken still carries (`text-davinci-003`, `code-cushman-001`, and friends) and the Anthropic, Google and xAI models their providers have shut down. Add `--include-retired` to see them. It affects `--list-models` and nothing else: on any other command it is accepted and ignored, and the retired names still count exactly as before.

```sh
toko --list-models --include-retired | grep davinci
```

```txt
openai/code-davinci-001
openai/code-davinci-002
openai/code-davinci-edit-001
openai/davinci
openai/davinci-002
openai/davinci-codex
openai/text-davinci-001
openai/text-davinci-002
openai/text-davinci-003
openai/text-davinci-edit-001
openai/text-search-davinci-doc-001
openai/text-similarity-davinci-001
```

## API keys and optional providers

Some providers require API credentials to access token counting endpoints:

- **Anthropic** – set `ANTHROPIC_API_KEY`
- **Google Gemini/Gemma** – set `GOOGLE_API_KEY`
- **xAI Grok** – set `XAI_API_KEY` for exact counts. It is the one provider that degrades instead of failing: without the key, Toko falls back to the Grok-1 tokenizer from `toko[transformers]` and marks the count approximate.

OpenAI needs no key at all; those counts are computed locally with `tiktoken`.

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
default_format = "text"    # one of: text, json, csv, tsv
respect_gitignore = true
auto_update_prices = false # fetch latest pricing when cached data is stale

[toko.exclude]
patterns = ["*.log", "*.tmp", "**/__pycache__/*"]

[toko.api_keys]
anthropic = "sk-ant-..."
xai = "xai-..."
# An "openai" entry is accepted and exported as OPENAI_API_KEY, but counting never
# reads it: OpenAI models are tokenized locally with tiktoken.
```

Config values act as defaults; command-line flags always win.

## Teaching Toko about a new model

The model registry is data, not code. Toko ships it as `models.toml` inside the
package and merges `$XDG_CONFIG_HOME/toko/models.toml` (defaults to
`~/.config/toko/models.toml`) over the top, so a model that launches today can
be counted today:

```toml
[[model]]
name = "claude-fable-6"
provider = "anthropic"
tokenizer = "claude-opus-4-7" # which Anthropic tokenizer generation it uses

[[model]]
name = "gpt-6.1"
provider = "openai"
encoding = "o200k_base" # optional; unknown OpenAI models are estimated
```

Entries merge field by field and yours win, so naming an existing model
overrides just the fields you list and leaves the rest of the shipped entry
alone. Available fields are `name`, `provider`, `encoding`, `api_endpoint`,
`retired`, `redirects_to`, `tokenizer`, `listed`, and `aliases` (Google and xAI
only). `aliases` is the one field that accumulates rather than replaces: your
list is appended to the shipped one, so adding a nickname cannot quietly
unpublish the alternate names a model already answered to. The trade-off is that
an alias can never be removed, only re-pointed by declaring it on another model,
and the model declared last in the **merged** registry wins.

Merged order is not the order of your file. An entry that overrides a shipped
model keeps that model's position in the packaged registry, so reordering your
own file changes nothing. To re-point an alias so that it always takes effect,
declare it on a model name the packaged registry does not have — brand-new names
are appended after everything else:

```toml
[[model]]
name = "my-gemini"            # a new name, so this entry lands last and keeps the alias
provider = "google"
aliases = ["gemini-exp-1206"]
```

Toko says so on stderr when a declaration you made loses the alias to a model
declared later, and when an alias repeats the name of a real model (model names
are matched first, so such an alias could never fire). A re-point that works is
silent. Model and alias names are both matched case-insensitively. A malformed
file — including one that is not valid UTF-8 — is reported on stderr and skipped,
and Toko falls back to the registry it shipped with rather than failing to run.

`tokenizer` matters for Anthropic: Claude Opus 4.7 changed tokenizer, and the
same text counts roughly 30% higher on 4.7-generation models. Toko refuses to
resolve a shorthand name that would span the two, so set it correctly.

## Caching and pricing data

- Counts are cached in `$XDG_CACHE_HOME/toko/token_cache.db`.
- Pricing starts from the data bundled inside the `genai-prices` dependency. Refreshed pricing is never written back into the installed package: it lands in the same cache directory as the counts, as `$XDG_CACHE_HOME/toko/prices.json`, and is loaded from there on later runs. When `auto_update_prices` is `true`, Toko silently refreshes that file if it is older than a day. Fetch failures never abort your command.

## Development tasks

```sh
just lint          # Ruff check & format
just typecheck     # ty type checking
just test          # run "fast" tests
just check-all     # run the full set of checks
just update-readme # re-run the examples above and paste back what they print
```

## License

MIT
