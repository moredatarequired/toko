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
- The public Python API, which is exactly the three names in `src/toko/__init__.py`'s
  `__all__`: `count_tokens`, `TokenCount`, and `__version__`.

Everything else is internal and can change in any release without notice. That is
every module under `toko.` beyond those re-exports, the non-underscored names those
modules define, and `toko.models.MODELS`, which is a mutable module global. Python
will happily let you import from a submodule; that is not the same as a promise.

None of this is a commitment yet. Issue
[#28](https://github.com/moredatarequired/toko/issues/28) is the checklist gating
1.0, and the surfaces listed above are the ones it is still reshaping.

## What is not covered, at 1.0 or now

Model coverage and pricing data are outside the versioned surface.

Models come from four places that move independently of a toko release: the packaged
registry, `MODEL_TO_ENCODING` read out of the installed tiktoken (so upgrading
tiktoken alone can change `--list-models`), whichever optional tokenizer extras are
installed, and any `[[model]]` entries in your own config. Adding, hiding or
re-pointing a model is a normal `feat` or `fix`, never a major-bump break.

Prices move without any toko release at all. They start from the data bundled inside
`genai-prices` and refresh from that project's upstream feed into
`$XDG_CACHE_HOME/toko/prices.json` once you run `toko update-prices` or turn on
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

- A model the registry marks `retired`, which today is Anthropic, Google and xAI
  entries, warns on stderr when you count it. The notice gives the retirement date,
  or says the date is unpublished, and then either says the provider will reject or
  redirect the name, or, when the registry knows the replacement, says outright that
  the number you are reading is the replacement's count and not the one you asked
  for.
- The OpenAI names in `RETIRED_OPENAI_MODELS` are silent. They still tokenize
  locally and still return a count; that list only hides them from `--list-models`.
- `--include-retired` puts both kinds back into the `--list-models` output and does
  nothing else, as its help text says.
- Nothing fails. There is no flag that turns a retired model into an error.

Whether a retired name should fail instead, or should redirect and report that in
the structured output rather than only on stderr, is unresolved and tracked in #28.
