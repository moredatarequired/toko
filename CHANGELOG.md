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
