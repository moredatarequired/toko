# Agent Handbook

## Purpose & Scope

- This is `toko`, a CLI-first token counting tool for LLMs.
- The repository is hosted at https://github.com/moredatarequired/toko and uses GitHub Actions for CI.
- Core functionality: count tokens for multiple LLM providers (OpenAI via tiktoken, Anthropic/Google/xAI via their APIs), support multiple input methods (stdin, files, directories, URLs), and provide flexible output formats.

## Python Conventions

- Python 3.14; managed with `uv`. Avoid `from __future__ import annotations` and other forward-compat hacks.
- No relative imports; always import from the fully qualified package (e.g., `from toko.counter import count_tokens`).
- Keep abstractions minimal—prefer straightforward, readable functions over clients/config wrappers unless strictly necessary.
- Use Pydantic models for complex classes or datastructures, and prefer them over `dataclasses.dataclass`.
- Do not add pointless or boilerplate docstrings or comments, and do not document obvious function parameters. Most of the time, the purpose and use of a function or class can be correctly inferred from its signature (names and type hints), and in these cases additional documentation makes the code harder to read, not easier. Only add comments when something is surprising, unintuitive, or has behavior too complex to capture through type hints and names.

## Formatting, Linting, Tests

- Run the linter after every significant change: `just lint` (runs `ruff check --fix` then `ruff format`).
- Type checking: `just typecheck` (invokes `ty`).
- Tests: `just test-all` (pytest).
- Tests should rely on minimal (preferably no) mocking. An integration test that actually calls an external API (or uses tiktoken) is much better and more useful than unit tests that mock dependencies.

## Running the CLI

- Use `uv run toko <args>` to run the CLI during development.
- Install hooks once with `just setup` (installs Lefthook git hooks).
- Environment variables (API keys) can be loaded via `~/.config/toko/config.toml` or environment variables like `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.

## Additional Practices

- Be mindful of Lefthook/ruff auto-fixes when editing—rerun lint if generated files change.
- When introducing new model support, focus on the minimal data needed (model name, encoding/API endpoint, cost data from genai-prices).
- Default behavior should respect .gitignore files (like ripgrep) and recurse into directories.

## Commit Style

- One-line commits are preferred for small, surgical changes.

- Use the imperative voice.

- Prefer [Conventional Commits](https://www.conventionalcommits.org/); we use `commitizen` to check but not enforce the format.

- Keep bodies concise. Complete sentences should be reserved for explaining very involved changes or non-obvious context.

- **DO NOT** include yourself as an author or co-author on commits or pull requests.

- Example (`feat(models): add gemini-2.0-flash support`) for a moderately complex change:

  ```
  feat(models): add gemini-2.0-flash support

  Add token counting for Gemini 2.0 Flash via Google API.
  - adds model registry entry for gemini-2.0-flash
  - implements count_tokens API call with caching
  - updates --list-models output
  ```

- Open Pull Requests with `gh pr create --title <title> --body <description>`.

- Use `gh pr list` and `gh issue list` for viewing PRs and issues.
