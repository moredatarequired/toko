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
since the most recent tag, so a checkout missing `0.2.1` will compute the wrong bump.

## 2. Bump

```sh
uv run cz bump
```

This does four things in one commit: works out the new version from the Conventional
Commit messages, writes it into `pyproject.toml`, regenerates `CHANGELOG.md`
(`update_changelog_on_bump = true`), and creates the tag.

`major_version_zero = true` keeps Toko pre-1.0, so a `BREAKING CHANGE` in the range
bumps the minor rather than the major. On the commits sitting on `main` today that
yields **0.3.0**, not 1.0.0. Confirm the plan before committing to it:

```sh
uv run cz bump --dry-run
```

Going to 1.0.0 is a deliberate act: drop `major_version_zero`, or pass an explicit
`--increment MAJOR`.

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
1. **Smoke test, wheel** — installs the built wheel with the `[all]` extras into an
   isolated environment and runs `tests/smoke_test.py`.
1. **Smoke test, sdist** — the same check against the `.tar.gz`, which is what catches a
   file that the wheel happens to carry but the sdist omits.
1. **Publish** — `uv publish` to PyPI.

The `[all]` extras are required, not incidental: `tests/smoke_test.py` asserts
`len(models_by_provider) > 4`, and the Mistral and HuggingFace-backed providers only
register when `mistral-common` and `transformers` are installed. A bare install would
fail that assertion.

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
