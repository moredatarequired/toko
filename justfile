set dotenv-load

[no-exit-message]
recipes:
    @just --choose

setup: uv-sync
    uv run --group dev lefthook install

uv-sync:
    uv sync --all-groups --all-extras

lint *args:
    uv run --group dev ruff check --fix {{args}}
    uv run --group dev ruff format {{args}}

test *args:
    uv run --group test pytest -m 'not slow' {{args}}

test-all *args:
    uv run --group test pytest {{args}}

test-coverage *args:
    uv run --group test pytest --cov --cov-branch {{args}}

typecheck *args:
    uv run --group dev ty check {{args}}

scan-secrets:
    uv run --group dev detect-secrets-hook --baseline .secrets.baseline

update-readme:
    uv run python scripts/update_readme_examples.py

update-secrets:
    uv run --group dev detect-secrets scan --baseline .secrets.baseline

check-all: uv-sync lint typecheck scan-secrets test-all
