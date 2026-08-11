"""Configuration file handling."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from toko.output_format import OutputFormat


@dataclass
class Config:
    """Toko configuration."""

    default_model: str = "gpt-5"
    respect_gitignore: bool = True
    default_format: OutputFormat = OutputFormat.TEXT
    exclude_patterns: list[str] = field(default_factory=list)
    # repr=False so a crash traceback rendering a Config never echoes secrets.
    api_keys: dict[str, str] = field(default_factory=dict, repr=False)
    auto_update_prices: bool = False


def get_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        config_dir = Path(config_home) / "toko"
    else:
        config_dir = Path.home() / ".config" / "toko"

    return config_dir / "config.toml"


def _parse_output_format(value: object, config_path: Path) -> OutputFormat:
    try:
        return OutputFormat(value)
    except ValueError as e:
        valid = ", ".join(fmt.value for fmt in OutputFormat)
        raise ValueError(
            f"Invalid default_format {value!r} in {config_path}"
            f" (expected one of: {valid})"
        ) from e


_TYPE_NAMES: dict[type, str] = {
    bool: "a boolean",
    dict: "a table",
    list: "a list",
    str: "a string",
}


def _require_type[T](
    value: object,
    expected: type[T],
    *,
    key: str,
    config_path: Path,
    show_value: bool = True,
) -> T:
    if isinstance(value, expected):
        return value
    # show_value=False for secrets: naming the offending key is enough, and the
    # message reaches the terminal.
    found = f" {value!r}" if show_value else ""
    raise ValueError(
        f"Invalid {key}{found} in {config_path} (expected {_TYPE_NAMES[expected]})"
    )


def _require_bool(value: object, *, key: str, config_path: Path) -> bool:
    # bool subclasses int, so this also accepts the 1/0 spellings, which loaded and
    # behaved correctly before these fields were validated.
    if isinstance(value, int):
        return bool(value)
    raise ValueError(
        f"Invalid {key} {value!r} in {config_path} (expected {_TYPE_NAMES[bool]})"
    )


def load_config() -> Config:
    """Load configuration from file.

    Returns:
        Config object with settings from file, or defaults if file doesn't exist

    Raises:
        ValueError: If config file is invalid
    """
    config_path = get_config_path()

    if not config_path.exists():
        return Config()

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        raise ValueError(f"Error reading config file {config_path}: {e}") from e

    # Extract toko section
    toko_config = _require_type(
        data.get("toko", {}), dict, key="toko", config_path=config_path
    )

    # Check for auto_update_prices from env var or config
    env_auto_update = os.environ.get("TOKO_AUTO_UPDATE_PRICES", "").lower() in (
        "true",
        "1",
        "yes",
    )
    config_auto_update = _require_bool(
        toko_config.get("auto_update_prices", False),
        key="auto_update_prices",
        config_path=config_path,
    )

    exclude = _require_type(
        toko_config.get("exclude", {}), dict, key="exclude", config_path=config_path
    )
    # A bare string here would be iterated character by character into pathspec, where
    # the lone "*" silently excludes every file, so the element types matter too.
    exclude_patterns = _require_type(
        exclude.get("patterns", []),
        list,
        key="exclude.patterns",
        config_path=config_path,
    )
    for index, pattern in enumerate(exclude_patterns):
        _require_type(
            pattern, str, key=f"exclude.patterns[{index}]", config_path=config_path
        )

    api_keys = _require_type(
        toko_config.get("api_keys", {}),
        dict,
        key="api_keys",
        config_path=config_path,
        show_value=False,
    )
    for name, key_value in api_keys.items():
        _require_type(
            key_value,
            str,
            key=f"api_keys.{name}",
            config_path=config_path,
            show_value=False,
        )

    # Build and return config
    return Config(
        default_model=_require_type(
            toko_config.get("default_model", "gpt-5"),
            str,
            key="default_model",
            config_path=config_path,
        ),
        respect_gitignore=_require_bool(
            toko_config.get("respect_gitignore", True),
            key="respect_gitignore",
            config_path=config_path,
        ),
        default_format=_parse_output_format(
            toko_config.get("default_format", "text"), config_path
        ),
        exclude_patterns=exclude_patterns,
        api_keys=api_keys,
        auto_update_prices=env_auto_update or config_auto_update,
    )


def apply_api_keys(config: Config) -> None:
    # Map config key names to environment variable names
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "xai": "XAI_API_KEY",
    }

    for key_name, env_var in key_map.items():
        if key_name in config.api_keys and not os.environ.get(env_var):
            os.environ[env_var] = config.api_keys[key_name]
