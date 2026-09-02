# Releasing Toko

Releases are cut by Commitizen and published by GitHub Actions. Pushing the tag is the
only irreversible step, and it is the step that publishes.

## 1. Start from a clean, up-to-date `main`

```sh
git checkout main
git pull
git fetch --tags origin
```

Fetching tags is not optional. Commitizen derives the next version from the commits
since the most recent release tag, so a checkout missing that tag will compute the
wrong bump.

## 2. Bump

```sh
uv run cz bump
```

This does four things in one commit: works out the new version from the Conventional
Commit messages, writes it into `pyproject.toml`, regenerates `CHANGELOG.md`
(`update_changelog_on_bump = true`), and creates the tag.

`major_version_zero = true` keeps Toko pre-1.0, so a `BREAKING CHANGE` in the range
bumps the minor rather than the major: the three `BREAKING CHANGE` footers in the 0.3.0
release took 0.2.1 to 0.3.0, not to 1.0.0. Confirm what the range computes before you run
the bump above:

```sh
uv run cz bump --dry-run
```

Going to 1.0.0 is a deliberate act: drop `major_version_zero`, or pass an explicit
`--increment MAJOR`.

The bump commit runs the pre-commit hooks, and lefthook's `format markdown` job
re-indents the `BREAKING CHANGE` continuation lines cz emits flush-left. That cancels
cz's de-indentation, `stage_fixed: true` re-stages, and the commit lands looking
unremarkable. It only cancels if `just setup` installed the hooks; without them the
changelog lands de-indented. Either way, `uv run cz changelog` never comes back clean
on `main`, so an empty diff is not a check that the changelog is current.

## 3. Push the commit, then the tag

```sh
git push origin main
git push origin "$(uv run cz version --project)"
```

The tag push is what releases. Nothing is published until it lands.

## 4. What CI then does

`.github/workflows/release.yml` triggers on any tag matching `[0-9]*`, and runs:

1. `uv build` — builds the wheel and the sdist into `dist/` (gitignored; never
   committed).
1. **Smoke test, wheel, no extras** — installs the bare wheel into an isolated
   environment and runs `tests/smoke_test.py`. This is the only step that can catch a
   module toko imports but never declares: under `[all]`, `transformers` pulls in
   `huggingface-hub`, which depends on `click`, so an undeclared `import click` would
   resolve there no matter what `pyproject.toml` says.
1. **Smoke test, wheel, all extras** — the same file with `[all]`, which is what
   reaches the optional tokenizers. Each check gates on `importlib.util.find_spec`, so
   the bare run above skips the Mistral and transformers counts rather than failing
   them, and the provider listing is asserted against 4 providers there and 9 here.
1. **Smoke test, sdist, all extras** — the same check against the `.tar.gz`, which is
   what catches a file that the wheel happens to carry but the sdist omits.
1. **Publish** — `uv publish` to PyPI.

A bare install lists exactly 4 providers, and either extra adds to that on its own.
`transformers` registers `llama`, `deepseek`, `qwen`, and `huggingface`, taking the count
to 8; `mistral-common` registers `mistral`, taking it to 5. `[all]`, which is what the two
all-extras steps install, gets both and lists 9.

The job sets `XDG_CACHE_HOME` under the workspace so both `[all]` steps share one
Hugging Face download. The transformers count is the one check allowed to skip on its
own: it fetches a tokenizer from the Hub, which refuses anonymous callers with 429 on
its own schedule, and a rate limit must not block a release. Every other Hub failure —
a gated repo, a missing model — still fails the job.

The checks themselves also run on every PR, via `tests/test_release_smoke.py`, so a
broken assertion surfaces there rather than here.

Publishing uses [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/), so
there is no API token to rotate. The workflow's `pypi` environment supplies the OIDC
identity and must exist under **Settings → Environments** with the matching publisher
configured on PyPI.

## 5. Verify

Check the workflow run finished green, then confirm the release is installable and that
the metadata and license actually shipped:

```sh
uv tool install "toko@$(uv run cz version --project)"
```

## If a release goes wrong

A PyPI version can never be reused, even after a yank. Fix forward: land the fix on
`main` and cut the next patch release. Delete the bad tag only if the publish step never
ran.
