## 0.4.0 (2026-09-01)

### Feat

- **models**: detect the devstral, magistral and voxtral families
- **models**: advertise an exactly-counting Mistral name
- **cli**: count files concurrently with --jobs
- **cli**: add --sort for the per-file row order
- **output**: drop the borders from the text tables

### Fix

- **models**: declare encodings for the canonical names tiktoken knew only by prefix
- **counter**: stop taking exact OpenAI counts from tiktoken's prefix table
- **counter**: keep warming a tokenizer from failing a run that would have worked
- **cache**: close cache connections and load tokenizers before the pool
- **formatters**: source the table size the way rich does
- **formatters**: stop TERM=dumb from shrinking tables to 80 columns
- **cost**: match Mistral release numbers in either spelling
- **cost**: resolve dated Mistral releases to their own price entries
- **cost**: price the Mistral families as themselves, not as mistral-small
- **models**: detect the codestral, ministral and pixtral families
- **hooks**: guard just scan-secrets against an empty file list
- **hooks**: pass files to detect-secrets so the scan actually runs
- **cli**: build each tokenizer once, and bound and settle the concurrent run
- **cli**: keep --sort out of a --total-only run
- **scripts**: set NO_COLOR, the variable rich actually reads
- **scripts**: isolate every command of an example, not just the first
- **scripts**: keep a stale example's command and reasons apart
- **scripts**: stop blaming toko for uv's own warnings

### Refactor

- **models**: keep the newly declared encodings off --list-models
- **formatters**: let rich resolve the table size itself
- **cli**: name the default row order input and make --sort path sort

## 0.3.0 (2026-08-12)

### BREAKING CHANGE

- count_tokens returns a TokenCount instead of a bare int, and
  piped single-model TSV output for an approximate count now emits a full row
  rather than a bare number.
- format_file_table() now raises ValueError for an
  unrecognized output_format instead of returning a text table. Callers
  passing anything outside text/json/csv/tsv must handle the error.
- format_output() and format_text() no longer accept the
  positional total_only argument. Callers such as
  format_output(results, "text", True) must drop it; the argument had no
  effect on any output.

### Feat

- **cli**: keep the approximate marker in piped single-model TSV
- **toko**: publish count_tokens and TokenCount as the package API
- **counter**: return a TokenCount instead of a bare int
- **cli**: add --include-retired to --list-models
- **openai**: estimate unknown OpenAI models with o200k_base
- **models**: warn when one document declares a model name twice
- **models**: move the model registry into a packaged models.toml
- **models**: warn on retired models instead of leaking provider errors
- **models**: refresh the Anthropic, Google and xAI model registries
- **openai**: estimate unknown OpenAI models with o200k_base

### Fix

- **mistral**: resolve model names without mistral-common's deprecated from_model
- **deps**: upgrade to typer 0.27 and drop the direct click import
- **formatters**: keep every distinct caveat in a model total
- **formatters**: emit the JSON approximate field only when a count is approximate
- **toko**: keep unknown package attributes a type error
- **counter**: make key redaction unraisable and keep connect errno
- **counter**: stop quoting transport errors and guard malformed bodies
- **counter**: stop leaking API keys in error messages
- **scripts**: rewrite only the README examples whose commands succeeded
- **scripts**: stop silently writing error output into the README
- **scripts**: fail loudly when the README example shell is missing
- **config**: redact the whole api_keys name, not just its last dot-segment
- **config**: redact secret key names and reject non-boolean integers
- **models**: re-retire babbage-002, davinci-002 and gpt-35-turbo
- **models**: keep the live babbage-002, davinci-002 and gpt-35-turbo listed
- **config**: stop echoing an api_keys value written as a bare string
- **cli**: dispatch subcommands when a global option comes first
- **config**: validate config types instead of raising tracebacks
- **formatters**: quote csv fields via csv.writer
- **file-reader**: skip .git during recursive walks
- **tests**: stop HubFailures losing errors the Hub coincided with
- **tests**: attribute collected failures so a Hub 429 can still skip
- **tests**: let real failures win over the Hub rate-limit skip
- **models**: name the model that actually keeps a contested alias
- **models**: stop the duplicate-alias warning firing on the success path
- **cli**: stop --total-only making --text output more verbose
- **formatters**: don't total an unpriced model to $0.000000
- **cli**: don't follow a specific path error with a generic one
- **config**: reject an invalid default_format instead of crashing
- **cli**: keep counting when a file in a directory is unreadable
- **cli**: reject unknown --format values without dumping a traceback
- **formatters**: make --total-only mean what it says, in every format
- **cli**: keep counting after an unreadable path
- **cli**: allow options after positional paths
- **prices**: reject price data we cannot read prices out of
- **prices**: discard a cached price payload we cannot use
- **prices**: write the price cache atomically and report a corrupt one
- **prices**: install cached prices on every run
- **cli**: reach the price subcommands and keep the prices they fetch
- **models**: make a re-pointed or capitalised alias behave as documented
- **models**: keep the retirement warning on a "-latest" alias
- **openai**: match tiktoken lookups case-insensitively
- **cost**: price the model the user asked for, not its canonical alias
- **counter**: do not cache approximate xAI token counts
- **models**: source the xAI registry from xAI's published lists
- **models**: let Google resolve its own "-latest" aliases
- **counter**: warn about a retired model even when its count is cached
- **models**: match a Google alias prefix only on a separator
- **models**: extend a model's shipped aliases instead of replacing them
- **models**: skip a models.toml that is not UTF-8 instead of crashing
- **models**: resolve Google aliases by longest prefix, not registry order
- **cost**: price the model the user asked for, not its canonical alias
- **counter**: raise ValueError on malformed Anthropic and Google responses
- **counter**: warn when an xAI count falls back to an approximate tokenizer
- **formatters**: include cost data in json output

### Refactor

- **cli**: annotate all TokoGroup ctx params Any and harden the smoke test
- **counter**: require api_key on \_describe_request_failure
- move OutputFormat to a leaf module
- **prices**: read the update URL from its exported constant
- **models**: derive the advertised OpenAI list from the verified encodings
- **counter**: key the warn-once registry by warning kind
- **counter**: type the provider handler table and stub the xai fallback in its cache test
- **models**: keep presentation out of retirement_notice

## 0.2.1 (2025-12-12)

### Feat

- **models**: add claude-opus-4-5 and gpt-5.x model support

## 0.2.0 (2025-10-27)

### Feat

- **models**: refresh provider registries and docs
- **cli**: show optional extras in model listing
- **cli**: add header controls and key warnings
- add SQLite-based token count caching
- add auto-update-prices configuration option
- add update-prices command
- add configuration file support
- add cost estimation with genai-prices
- add URL support for remote content
- add file and directory support with gitignore
- use rich for beautiful table output
- **xai**: add support for 8 xAI Grok models
- **format**: add table format for multiple models
- **api**: add Anthropic and Google token counting
- **models**: add all tiktoken-supported OpenAI models
- **cli**: wire up token counting with formatters
- **counter**: implement OpenAI token counting with tiktoken
- add basic CLI structure with typer

### Fix

- **counter**: silence llama tokenizer warnings
- **cache**: honor xdg cache directory
- **google**: update to models supporting countTokens
- **hooks**: remove prepare-commit-msg hook

### Refactor

- **core**: streamline counting flow
