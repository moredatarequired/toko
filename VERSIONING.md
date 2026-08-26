# Versioning Toko

Toko is pre-1.0: `pyproject.toml` carries `0.3.0` and the package ships as
`Development Status :: 4 - Beta`. A minor bump is allowed to break you, so pin to a
minor — `toko~=0.3.0`, or `toko>=0.3,<0.4` — and read the changelog before moving.

## How the number moves today

Versions follow `semver2`, and tags are the bare number with no `v` prefix.
Commitizen works out each bump from the Conventional Commit subjects since the
previous tag, so the type you write on a commit is the version decision:

| Subject                                               | Bump             | Changelog entry |
| ----------------------------------------------------- | ---------------- | --------------- |
| `BREAKING CHANGE`, or `type!:`                        | minor, see below | yes             |
| `feat:`                                               | minor            | yes             |
| `fix:`, `perf:`, `refactor:`                          | patch            | yes             |
| `docs:`, `chore:`, `test:`, `style:`, `build:`, `ci:` | none             | no              |

`major_version_zero = true` is what collapses the first two rows into one. While the
major is 0, a breaking change bumps the minor rather than the major, which is how
0.3.0 followed 0.2.1 across a release whose changelog opens with a
`BREAKING CHANGE` section. That leaves no digit to signal "this one breaks you", so
pre-1.0 the changelog is the whole warning.

A commit whose type is not in that table produces no changelog entry, and neither
does a subject with no type at all, which `cz check` rejects at commit time if you
installed the hooks and nothing catches if you did not. See
[CONTRIBUTING.md](CONTRIBUTING.md) for what does and does not enforce the format.

Reaching 1.0.0 is a deliberate act rather than something `cz bump` can arrive at on
its own; [RELEASING.md](RELEASING.md) covers the two ways to do it.

## What 1.0 will cover

Once 1.0 is stamped, breaking any of these requires a major bump:

- The CLI: which commands and flags exist, and what they mean.
- The machine-readable output: the JSON, CSV and TSV shapes.
- Exit codes.
- The public Python API, which is exactly the six names in `src/toko/__init__.py`'s
  `__all__`: `count_tokens`, `TokenCount`, `Caveat`, `CaveatKind`, `Retirement`, and
  `__version__`.

Everything else is internal and can change in any release without notice. That is
every module under `toko.` beyond those re-exports, the non-underscored names those
modules define, and `toko.models.MODELS`, which is a mutable module global. Python
will happily let you import from a submodule; that is not the same as a promise.

None of this is a commitment yet. Issue
[#28](https://github.com/moredatarequired/toko/issues/28) is the checklist gating
1.0, and the surfaces listed above are the ones it is still reshaping.

## What is not covered, at 1.0 or now

Model coverage and pricing data are outside the versioned surface.

Models come from four places. The packaged registry ships inside the wheel, so it
moves with a release; the other three move without one: `MODEL_TO_ENCODING` read out
of the installed tiktoken (so upgrading tiktoken alone can change `--list-models`),
whichever optional tokenizer extras are installed, and any `[[model]]` entries in
your own config. Adding, hiding or re-pointing a model is a normal `feat` or `fix`,
never a major-bump break.

Prices move without any toko release at all. They start from the data bundled inside
`genai-prices` and refresh from that project's upstream feed into `prices.json` in
toko's cache directory — `$XDG_CACHE_HOME/toko` when it is set, otherwise the
platform's own cache location — once you run `toko update-prices` or turn on
`auto_update_prices`, so two runs of the same toko version can quote different costs.
Cost output is an estimate that changes under you by design.

## Deprecation

Nothing in toko is deprecated today, so this is what will happen rather than what
has.

Anything deprecated keeps working. It warns when you use it, it is named in the
CHANGELOG entry for the release that deprecated it, and it is removed no earlier
than the next minor release after that one. Removal is a breaking change and follows
the rules above, so pre-1.0 a removal lands in a minor bump like everything else.

Warnings go to stderr and never to stdout, so a piped run stays clean: whatever toko
has to say about a count, only the count reaches `jq`. Counting warnings are also
deduplicated on the pair of warning kind and model name, so counting a directory
prints a given notice once per process rather than once per file. They are plain
writes to `sys.stderr` rather than `warnings.warn` or `logging`, so a library caller
can only redirect `sys.stderr` wholesale rather than filter or silence them one at a
time; changing that is on #28.

## Retired models are a separate mechanism

A retired model is not a deprecated flag and is not on the clock above. Providers
retire models on their own schedule, and toko learns about it when the registry is
updated.

What toko does today, in full:

- The CLI refuses to count with a retired model. Naming one exits `1` before any input
  is read, so no partial table is printed, and the error carries the retirement date
  and the replacement when the registry knows one:

  ```txt
  Error: model 'grok-3' is retired (2026-05-15); it redirects to grok-4.3. Pass --include-retired to count with it anyway.
  ```

  A retirement whose date the provider never published reads `(date unknown)` where the
  date would be — `toko -m grok-3-mini --text "hello world"` prints
  `Error: model 'grok-3-mini' is retired (date unknown). Pass --include-retired to count with it anyway.`

- Two sources feed that gate, and a name from either is refused the same way. Registry
  entries marked `retired` — today Anthropic, Google and xAI names — carry a date and
  sometimes a redirect target. The OpenAI engines in `RETIRED_OPENAI_MODELS` carry the
  shutdown date OpenAI published and never a redirect, because a shut-down engine has
  nothing to redirect to; `-m text-davinci-003` is refused exactly like `-m grok-3`.

- Hidden from `--list-models` is not the same thing as refused, and the two OpenAI sets
  are no longer one list. The listing filter is `UNLISTED_OPENAI_MODELS`, which is
  `RETIRED_OPENAI_MODELS` plus live-but-unadvertised tiktoken names — `babbage-002`,
  `davinci-002`, `gpt-35-turbo`, `gpt-3.5`, `gpt2` and `gpt-2`. The gate reads only
  `RETIRED_OPENAI_MODELS`, so those six are kept out of the default listing and still
  count with no flag, no warning and exit `0`.

- `--include-retired` is what lets the count happen; putting the hidden names back into
  `--list-models` is its second effect rather than its only one, as its help text now
  says: "Count with retired models instead of failing, and list them in `--list-models`
  output". With the flag the count runs and the retirement moves to stderr —
  `Warning: grok-3 was retired on 2026-05-15; xai still answers for it but serves grok-4.3, so this count is grok-4.3's, not grok-3's.` —
  and `--format json` reports it in structured output too, as a `retirement` object of
  `model`, `date` and `redirects_to` on each count and total.

- The refusal is the CLI's, not the library's. `count_tokens` counts a retired model
  with no flag to pass, writes that same stderr warning, and reports it in
  `TokenCount.retirement`.

Whether a retired name should fail, redirect silently, or only warn was the open
question here, and failing is the answer toko settled on: a count taken under a retired
name is either another model's number or nothing at all, so the CLI declines to print
it unless you ask for it by name. What remains unsettled is the warning channel — see
the note above about plain `sys.stderr` writes, which is on #28.
