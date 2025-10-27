"""Configuration file handling."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Toko configuration."""

    default_model: str = "gpt-5"
    respect_gitignore: bool = True
    default_format: str = "text"
    exclude_patterns: list[str] = field(default_factory=list)
    api_keys: dict[str, str] = field(default_factory=dict)
    auto_update_prices: bool = False


def get_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        config_dir = Path(config_home) / "toko"
    else:
        config_dir = Path.home() / ".config" / "toko"

    return config_dir / "config.toml"


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
    toko_config = data.get("toko", {})

    # Check for auto_update_prices from env var or config
    auto_update = os.environ.get("TOKO_AUTO_UPDATE_PRICES", "").lower() in (
        "true",
        "1",
        "yes",
    )
    if not auto_update:
        auto_update = toko_config.get("auto_update_prices", False)

    # Build and return config
    return Config(
        default_model=toko_config.get("default_model", "gpt-5"),
        respect_gitignore=toko_config.get("respect_gitignore", True),
        default_format=toko_config.get("default_format", "text"),
        exclude_patterns=toko_config.get("exclude", {}).get("patterns", []),
        api_keys=toko_config.get("api_keys", {}),
        auto_update_prices=auto_update,
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
