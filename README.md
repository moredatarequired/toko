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
File     gpt-5
LICENSE    223
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
model	tokens	cost	approximate
gpt-5-mini	4	0.000001	false
claude-opus-4-5	11	0.000055	false
```

Costs come from the bundled `genai-prices` feed. A model it has no price for shows `N/A`
in the text table, an empty cell in CSV and TSV, and `null` in JSON.

### Work with directories, URLs, and filters

```sh
toko --exclude '**/__pycache__/*' src/
```

```txt
File                                   gpt-5
src/toko/__init__.py                     378
src/toko/cache.py                        863
src/toko/cli.py                        5,112
src/toko/config.py                     1,374
src/toko/cost.py                       1,387
src/toko/counter.py                    6,495
src/toko/data/__init__.py                  8
src/toko/data/models.toml              2,952
src/toko/data/openrouter_models.json     359
src/toko/file_reader.py                1,170
src/toko/formatters.py                 3,999
src/toko/models.py                     7,175
src/toko/output_format.py                 53
src/toko/price_update.py               1,414
src/toko/py.typed                          0
src/toko/result.py                       107
src/toko/sort_order.py                    50
TOTAL                                 32,896
```

- Directories are processed recursively by default and honor `.gitignore`.
- Use `--no-recursive` to stay shallow and `--no-ignore` to include ignored files.
- Counting files runs eight counts at a time, which matters most for the API-backed providers. Pass `--jobs`/`-j` (1 to 64) to change that, or `--jobs 1` to count one at a time. It has no effect on `--text` or piped stdin, and URLs are still fetched one after another.
- Use `--sort count` to put the biggest files first, which is the quick way to find
  whatever is filling a context window. `--sort path` sorts the rows by path instead.
- The default is `--sort input`, which leaves the rows in the order the paths were read.

`--sort path` compares the paths as the `File` column shows them, so files in the same
directory stay together. A directory scan already arrives in near-path order, but not the
same one: the scan compares path components, so `src/toko.py` lands after everything under
`src/toko/`, where `--sort path` puts it first. `--sort count` ranks rows by the leftmost
model column, which is always your first `--model`: model columns run in the order you
named the models in, and a model no file could be counted for keeps its column rather
than dropping out of that order. Files the model has no count for sort last, after every
file that has one, because no count is not a count of zero; within each group the path
breaks the tie. Those cells read `N/A` in the text table and are empty in CSV and TSV. The
`TOTAL` row stays at the bottom whichever order you pick, and the order applies to every
format, not just the text table, so `--sort count --format json` lists its sources in the
same order the table would show. Runs that count `--text` or stdin have no file rows, so
`--sort` is accepted and ignored there.

## Machine-readable output

Toko can emit structured output without post-processing.

```sh
toko --model gpt-5 --format json LICENSE
```

```json
{
  "schema_version": 1,
  "results": [
    {
      "source": {
        "kind": "file",
        "name": "LICENSE"
      },
      "counts": [
        {
          "model": "gpt-5",
          "tokens": 223,
          "approximate": false,
          "cost": null,
          "caveats": [],
          "retirement": null,
          "reason": null
        }
      ]
    }
  ],
  "totals": [
    {
      "model": "gpt-5",
      "tokens": 223,
      "approximate": false,
      "cost": null,
      "caveats": [],
      "retirement": null,
      "reason": null
    }
  ]
}
```

Every JSON run emits that one document, whatever flags it was given:

- `schema_version` is `1`. It changes only when the shape below does.
- `results` holds one entry per source counted, each naming where the counts came
  from — `{"kind": "text", "name": null}` for `--text` or stdin, `"file"` with the
  path, `"url"` with the URL — and `totals` holds one count per model, summed across
  the sources. Both keys are always present and always arrays. `--total-only` empties
  `results` rather than changing the document's shape, and a `--text` run's totals
  simply repeat its single set of counts.
- A count object always carries all seven of `model`, `tokens`, `approximate`, `cost`,
  `caveats`, `retirement` and `reason`. No key appears or disappears with a flag or
  with what the counting produced: `cost` is `null` without `--cost`, `caveats` is `[]`
  when the count is the model's own, and `retirement` is `null` for a live model.
  `jq '.totals[] | .tokens'` reads any run that printed a document.
- `tokens` is `null` exactly when the model could not be counted, and `reason` then
  holds the failure — the same sentence the warning on stderr carries. On a count that
  succeeded, `reason` is `null`; on one that did not, `approximate` and `cost` are
  `null` too, since there is no count for them to describe. `reason` is a human-readable
  explanation and explicitly **not** machine-parseable: branch on `tokens == null`, not
  on its wording. It is JSON-only, like `caveats`, because a delimited cell cannot carry
  a free-text sentence.
- Every count array in a document — each `results[].counts` and `totals` — holds one
  entry per model you named with `--model`, in that order, whether or not that model
  could be counted for that source. Which models a document describes is a function of
  the command, not of what the counting managed to produce, so a run against one
  directory and the same run against another describe the same models. `totals` still
  sums only the sources that produced a count, and is `null` for a model none did.
  **Match on `model` anyway; never index or zip the arrays together.**

```sh
toko --header --model gpt-5 --format csv --text "hello world"
```

```csv
model,tokens,approximate
gpt-5,2,false
```

Use `--format tsv` to force TSV even when running interactively.

CSV and TSV have one shape per command, and you can work it out before you run it. A
`--text` or stdin run is model-major: a `model` column, a `tokens` column, a `cost`
column if and only if you passed `--cost`, and an `approximate` column, then one row per
model you named. A run over paths is file-major: a `file` column, then `<model>_tokens`,
`<model>_cost` (with `--cost`) and `<model>_approximate` for each model in the order you
named them, then one row per source — or, with `--total-only`, one `TOTAL` row in place
of them. A model named twice is asked for once on either shape, so `-m gpt-5 -m gpt-5`
emits what `-m gpt-5` emits. Nothing in that depends on what the counting produced: a
model that failed everywhere keeps its columns, and a run in which every count failed
prints the same header and the same rows with the cells empty. `--no-header` removes the
header row and nothing else. The one command that emits something other than the two
shapes above is a headerless single-model TSV `--text` or stdin run, which collapses to
a bare number — and that too is decided by the command rather than by the counting, as
the next section describes.

The text table is the one written for people, and it keeps the `$`; CSV and TSV write
the `cost` column for a program instead:

- **A bare number, never a currency symbol** — no `$`, and no rounding to `0.000000`
  for a fraction of a cent. `float()` accepts every non-empty cell toko writes in a
  `cost` column; the empty ones are described in the last bullet.
- **Always positional decimal, never an exponent.** A fraction of a cent is written
  `0.00000375`, not `3.75e-06`, so `sort -n` orders a cost column correctly and `bc`
  can read a cell at all — both of which mis-handle exponent form, `sort -n` silently.
- **The same number as `--format json` reports.** A cost is rounded to twelve
  significant digits once, where it is produced, so the delimited cell and the JSON
  number are one value written two ways. Where there is no cost the two say so
  differently: JSON writes `null`, the cell is empty.
- **An empty cell for a model with no price, and for one with no count.** No number
  would be honest in either — `0` reads as free, and as zero tokens — so the cell holds
  nothing; where it is the count that is missing, the `<model>_tokens` and
  `<model>_approximate` cells are empty too, and `--format json` says why under
  `reason`. Beware that this is an *empty field*, not a missing one: `awk -F'\t'` and
  `awk -F,` see it correctly, but bare `awk` splits on runs of whitespace and collapses
  the empty cell away, shifting every field after it left. Since headerless TSV is what
  a piped run emits by default, pass `-F'\t'`.

### What a piped run emits without `--format`

Since a non-TTY stdout already defaults to headerless TSV, a piped or captured run needs
no `--format` at all. A single-model `--text` or stdin run collapses one step further,
to a bare number, which is what makes the usual scripting shape work:

```sh
n=$(toko -m gpt-5 --text "hello world")   # 2
```

What decides that shape is the command, never the counting: one model named and a
non-TTY stdout is a bare number whatever comes back, including nothing — a model that
failed prints an empty line, and the run exits `1`. Two things are worth knowing before
you parse the output:

- The collapse to a bare number is for `--text` and stdin only. Give `toko` a path and
  you get the file-major shape above, a `file<TAB>tokens<TAB>approximate` row per file:
  the filename has to go somewhere, and a run over paths keeps the columns its shape
  always has rather than dropping them for having only one model to describe.
- A bare number has nowhere to say that a count is **approximate**, and the marker is
  on stderr, which the process at the other end of the pipe is not reading. Ask for a
  shape that has somewhere to put it if a run of yours can hit any of the three paths
  that produce an approximate count — they are listed under **Library usage** below.
  `--header` is enough: `toko --header --format tsv -m gpt-6 --text "hello world"`
  writes `model<TAB>tokens<TAB>approximate` and then `gpt-6<TAB>2<TAB>true`. So is
  `--format json`, where every count carries `approximate` of its own.

## Exit codes

| Code | Meaning                                                                                                                                                            |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `0`  | Toko printed counts. Some of what you asked for may still have failed — see the inconsistency below.                                                               |
| `1`  | The run produced no usable result, or one of its inputs did not survive. **Output may still have been printed, and it is short** — see the warning below the list. |
| `2`  | The command line itself was wrong: an unknown option, a missing value, or an unsupported `--format`. Typer reports these before Toko runs.                         |

A run exits `1` when:

- **no input was given _at a terminal_** — no `--text`, no paths, and stdin still the
  tty. Redirected or piped stdin is input, however little of it there is:
  `toko -m gpt-5 < /dev/null` counts the empty string, prints `0` and exits `0`;
- no files matched, which includes a directory whose every file was skipped;
- **a path could not be found, or a file could not be read** — even when other paths
  were counted. Toko resolves the paths before it counts anything, so the error reaches
  stderr first and the counts follow it, at a terminal and in a pipe alike. A file that
  is not valid UTF-8 is the exception: it is skipped with a `Warning: Skipping binary file …` on stderr, it contributes no row and no tokens, and the *skip itself* does not
  fail the run. It is only the skip that is exempt, though. If skipping leaves nothing
  to count — a binary file named on its own, or a directory of them — the run still
  exits `1` under the "no files matched" rule above: `toko -m gpt-5 photo.png` warns,
  then prints `Error: No files found matching criteria` and exits `1`. A binary file
  alongside a file that did count exits `0`;
- every model failed for every input;
- a **retired model** was named without `--include-retired` (nothing is read or
  counted in that case; the error names the model, its retirement date and its
  redirect target, if it has one);
- the config file is unreadable, or `toko update-prices` could not fetch prices.

> **A nonzero exit means the totals cover only what succeeded.** Toko prints the
> results for the inputs it could read *before* it exits `1`, so a partial failure
> still emits a complete, well-formed document — a full JSON envelope with both
> `results` and `totals`, or a full table, of exactly the shape a successful run of the
> same command emits. A model that failed is marked, by a `null` `tokens` with a
> `reason` in JSON and by empty cells in CSV and TSV. An *input* that failed is not:
> the totals simply sum a smaller set of files than a successful run would, and no
> field says so. Check the exit code before you trust a total, and read stderr for
> which inputs are missing.

One known inconsistency, which this table describes rather than hides: a model that
fails among several — a missing `ANTHROPIC_API_KEY`, say — leaves a warning on stderr,
keeps its column with every cell empty, and still exits `0`, while a *path* that fails
among several exits `1`. Making the two agree is breaking change 4 of
[issue #28](https://github.com/moredatarequired/toko/issues/28); until it lands, check
stderr, or count one model per invocation if a partial result would be dangerous.

## Library usage

`count_tokens` is the whole public API, and it returns a `TokenCount` describing how the
number was reached:

```python
from toko import count_tokens

result = count_tokens("hello world", model="gpt-5")
print(result.count)  # 2
```

| Field         | Type                 | Meaning                                                                                        |
| ------------- | -------------------- | ---------------------------------------------------------------------------------------------- |
| `count`       | `int`                | The token count.                                                                               |
| `model`       | `str`                | The canonical model the request resolved to, which need not be the string you passed.          |
| `provider`    | `str`                | The provider that produced the count.                                                          |
| `approximate` | `bool`               | True when the count came from a stand-in tokenizer rather than the model's own.                |
| `caveats`     | `tuple[Caveat, ...]` | Why the count is approximate, one `Caveat` per reason; empty when there is nothing to say.     |
| `cost`        | `float \| None`      | Estimated cost in USD, when pricing data covers the model.                                     |
| `retirement`  | `Retirement \| None` | Set when the named model has been retired, with the date and what the provider serves instead. |

An unreachable tokenizer is usually an error, not a fallback: `count_tokens` raises
`ValueError` when `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY` is missing, and when no
provider can be detected from the name at all. Exactly three paths return an approximate
count instead, so check `approximate` on those before treating a count as exact:

- An OpenAI name neither Toko nor `tiktoken` lists — one that still reads as an OpenAI
  model, such as `gpt-6` — is counted with the encoding `tiktoken` maps its family to
  when the name starts with a family it knows (`gpt-4-1` gets `cl100k_base` through
  `gpt-4-`), and otherwise with `o200k_base`, the encoding every OpenAI model since
  `gpt-4o` uses. A family prefix is a good guess at the encoding but no evidence the
  name exists, so either way the count is an estimate and the `Caveat` it carries names
  the encoding the count came from.
- An xAI model is counted with the Grok-1 Hugging Face tokenizer when `XAI_API_KEY` is
  unset or the xAI endpoint fails. That stand-in needs `toko[transformers]`; without it
  the call raises instead.
- A Mistral name Toko has no pinned tokenizer for — a rolling alias such as
  `mistral-large-latest`, or a release newer than the tokenizers `mistral-common` ships —
  is counted with the bundled tekken tokenizer, whose vocabulary every Mistral model
  since Nemo shares. An exact count needs a release `mistral-common` bundles a tokenizer
  for, and those stop at November 2024: `mistral-large-2411` counts exactly,
  `mistral-medium-2506` cannot.

```python
from toko import CaveatKind, count_tokens

result = count_tokens("hello world", model="grok-4.5")
for caveat in result.caveats:
    if caveat.kind is CaveatKind.XAI_GROK1_STANDIN:
        print(f"stood in with {caveat.tokenizer}, because {caveat.reason}")
    print(caveat.message)  # the sentence Toko printed on stderr
```

Each `Caveat` carries a `kind` (a `CaveatKind`: `OPENAI_ENCODING_GUESS`,
`XAI_GROK1_STANDIN` or `MISTRAL_TOKENIZER_FALLBACK`), the `model` it is about, the
`message` a person should read, and whichever of `encoding`, `tokenizer` and `reason`
that kind has to report. Branch on `kind` and read the fields; `message` is prose and
its wording may change, so it is the one part not to parse.

A `Retirement` is separate from the caveats on purpose: a caveat says the tokenizer was
substituted, while `retirement` says the *model* is gone. It carries the `model`, the
`date` it was retired (`None` when the provider published none), and `redirects_to`,
the model the provider serves in its place. The CLI refuses a retired model unless
`--include-retired` is passed; the library counts it and describes it in this field.

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

Retired models are hidden from that listing: both the dead OpenAI engines tiktoken still carries (`text-davinci-003`, `code-cushman-001`, and friends) and the Anthropic, Google and xAI models their providers have shut down. A few live names tiktoken carries are hidden too, because listing them would mislead rather than help: `gpt-35-turbo` (the Azure deployment spelling of `gpt-3.5-turbo`), the bare family name `gpt-3.5`, `babbage-002` and `davinci-002` (shutdown 2026-09-28), and `gpt2`/`gpt-2` (open weights, never OpenAI API models). Add `--include-retired` to see all of them. Being hidden is not being refused: the live names above count normally without any flag. Naming a genuinely retired model without the flag is an error, and the run stops before anything is read:

```sh
toko -m grok-3 --text "hello world"
# Error: model 'grok-3' is retired (2026-05-15); it redirects to grok-4.3. Pass --include-retired to count with it anyway.
```

That is the point of the flag: `grok-3` still answers, but what answers is `grok-4.3`, so the number and its price belong to a model you did not ask for. With `--include-retired` the count happens, the warning stays on stderr, and JSON reports the `retirement` object alongside the count.

The refusal follows the name rather than one spelling of it: surrounding whitespace, a router path ending in it (`openrouter/xai/grok-3`), and a `-latest` alias of a retired model (`grok-3-latest`) are all refused too. **A prefix is dropped only when every segment of it is empty or is one of a fixed set of routing segments — `openai/`, `anthropic/`, `xai/`, `mistral/`, `llama/`, `deepseek/`, `qwen/`, and `openrouter/` — because such a prefix is addressing rather than a repo owner. So `openrouter/text-davinci-003` and `/text-davinci-003` are the shut-down engine and are refused, while a Hugging Face repo whose last segment happens to be a retired name keeps its prefix and counts: `Xenova/text-davinci-003` and `openai-community/gpt2` are live repos, and calling either retired would be false about them.**

`google/` and `huggingface/` are left out of that set, because both are Hub organisations that own real repos, so `google/gemma-3-1b-it` stays that repo rather than becoming Google's `gemma-3-1b-it`. That exemption wins over the spelling `--list-models` prints, so a `google/` prefix keeps any Gemini name away from Google — retirement has nothing to do with it. `xai/grok-3` is refused as retired, but `google/gemini-2.0-flash-001` is not: it is read as a Hub repo id, and since no such repo answers, the run fails with `not found on HuggingFace` rather than with the retirement error. The live `google/gemini-2.5-flash` fails exactly the same way. Drop the `google/` prefix and the name reaches Google, where a retired one earns the retirement refusal and a live one is counted.

Whatever the gate matched on is what the run reports, so `--include-retired` gives `anthropic/curie` the same `retirement` object and stderr warning that bare `curie` gets.

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
- **Mistral** – install `toko[mistral]`; no API key is required for offline tokenization. `mistral-common` bundles the tokenizers up to Mistral Large 2411, so counts for newer models and for `-latest` aliases are marked approximate. Once the extra is installed, `--list-models` shows the Mistral names, including `mistral-large-2411` for an exact count.

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

## Stability

Toko is pre-1.0, and while the major is 0 a minor bump is allowed to break you:
breaking changes land in the minor rather than the major. Pin to a minor
(`toko~=0.3.0`) and read the changelog before moving. [VERSIONING.md](VERSIONING.md)
covers what 1.0 will commit to, what stays outside it, and how deprecations are
handled.

## License

MIT
