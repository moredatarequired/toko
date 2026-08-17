# Contributing to Toko

## Setup

```sh
git clone https://github.com/moredatarequired/toko
cd toko
just setup
```

`just setup` runs `uv sync --all-groups --all-extras` and then installs the Lefthook
hooks. That second step is the one that matters: the hooks are the only thing that
runs any check locally, so without `just setup` nothing lints, nothing type checks,
no test runs on commit, and your commit messages go unvalidated. The first sign of a
problem is then a red PR, except for the commit message, which CI never checks.

Toko requires Python 3.14 (`requires-python = ">=3.14"`), which uv installs and
manages for you.

## Running it

```sh
uv run toko "some text"
uv run --env-file .env toko README.md
```

The second form is for counting against a provider API that needs a key. Copy
`.env.example` to `.env` and fill in what you need; keys can also live under
`[toko.api_keys]` in `~/.config/toko/config.toml`.

## Checks

```sh
just lint       # ruff check --fix, then ruff format
just typecheck  # ty
just test       # fast tests only (-m 'not slow')
just test-all   # every test
just check-all  # sync, lint, typecheck, scan-secrets, test-all
```

`just check-all` is the one to run before pushing. Tests that need a provider API key
skip rather than fail, so the whole suite passes on a checkout with no keys
configured at all; CI runs it exactly that way.

## What the pre-commit hook does

`lefthook.yml` runs eight jobs in parallel on every commit: ruff through `just lint`,
ty through `just typecheck`, a `git diff --check` for conflict markers and whitespace
damage, the fast test suite, `uv-sort` over `pyproject.toml`, codespell over Python
and Markdown, mdformat over Markdown, and detect-secrets against `.secrets.baseline`.

Three of those rewrite files and re-stage them, so a commit can come back holding
more than you wrote: ruff-formatted Python, a re-sorted dependency list, or Markdown
reflowed to mdformat's shape. That is working as intended. Write Markdown in whatever
shape reads well and let the hook settle the formatting.

## Commits

Conventional Commits, imperative voice, one line for a small surgical change.

Enforcement is thinner than it looks. `cz check` runs as a local `commit-msg` hook,
so it only exists if you ran `just setup`, and CI does not check commit messages at
all: the `Code Quality Checks` workflow runs `lefthook run pre-commit`, which never
reaches the commit-msg stage.

Get the format right anyway, because the CHANGELOG is generated from these subjects
at release time. A subject that is not `feat`, `fix`, `perf`, `refactor` or a
breaking change produces no changelog entry, and one with no type at all produces
nothing. [VERSIONING.md](VERSIONING.md) has the full type-to-bump table.

Do not credit yourself as an author or co-author on a commit or a pull request.

## Pull requests

Two workflows run on every PR: `Code Quality Checks` runs the entire pre-commit set
over all files, and `Tests` runs `just test-all` on a single interpreter.

## Conventions

Prefer a real integration test to a mocked one. A test that actually calls the
provider API or actually runs tiktoken is worth more than one asserting against a
mock, and the suite is built to skip cleanly when a key is missing rather than force
you into mocks.

Import rules, type-hint conventions and the house line on comments are in
[AGENTS.md](AGENTS.md). Cutting a release is [RELEASING.md](RELEASING.md).
