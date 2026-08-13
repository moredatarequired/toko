"""Tests for config file loading."""

import pytest

from toko.config import load_config
from toko.output_format import OutputFormat

SENTINEL = "sk-ant-FAKE-SENTINEL-XYZZY"


@pytest.fixture
def write_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("TOKO_AUTO_UPDATE_PRICES", raising=False)
    config_dir = tmp_path / "toko"
    config_dir.mkdir()

    def write(contents: str):
        (config_dir / "config.toml").write_text(contents)

    return write


def test_missing_file_returns_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = load_config()
    assert config.default_model == "gpt-5"
    assert config.respect_gitignore is True
    assert config.default_format == OutputFormat.TEXT
    assert config.exclude_patterns == []
    assert config.api_keys == {}


def test_full_config_loads(write_config):
    write_config(
        "[toko]\n"
        'default_model = "gpt-4.1"\n'
        'default_format = "csv"\n'
        "respect_gitignore = false\n"
        "auto_update_prices = true\n"
        "[toko.exclude]\n"
        'patterns = ["*.log"]\n'
        "[toko.api_keys]\n"
        f'anthropic = "{SENTINEL}"\n'
    )

    config = load_config()
    assert config.default_model == "gpt-4.1"
    assert config.default_format == OutputFormat.CSV
    assert config.respect_gitignore is False
    assert config.auto_update_prices is True
    assert config.exclude_patterns == ["*.log"]
    assert config.api_keys == {"anthropic": SENTINEL}


def test_non_table_toko_section_is_rejected(write_config):
    write_config('toko = "oops"\n')
    with pytest.raises(ValueError, match=r"Invalid toko 'oops' in .*expected a table"):
        load_config()


def test_exclude_as_a_list_is_rejected(write_config):
    write_config('[toko]\nexclude = ["*.log"]\n')
    with pytest.raises(ValueError, match=r"Invalid exclude .*expected a table"):
        load_config()


def test_non_string_default_model_is_rejected(write_config):
    write_config("[toko]\ndefault_model = 5\n")
    with pytest.raises(
        ValueError, match=r"Invalid default_model 5 in .*expected a string"
    ):
        load_config()


def test_string_respect_gitignore_is_rejected(write_config):
    write_config('[toko]\nrespect_gitignore = "false"\n')
    with pytest.raises(
        ValueError, match=r"Invalid respect_gitignore 'false' in .*expected a boolean"
    ):
        load_config()


def test_string_auto_update_prices_is_rejected(write_config):
    write_config('[toko]\nauto_update_prices = "yes"\n')
    with pytest.raises(
        ValueError, match=r"Invalid auto_update_prices 'yes' in .*expected a boolean"
    ):
        load_config()


def test_env_var_enables_auto_update(write_config, monkeypatch):
    write_config("[toko]\n")
    monkeypatch.setenv("TOKO_AUTO_UPDATE_PRICES", "1")
    assert load_config().auto_update_prices is True


def test_non_table_api_keys_is_rejected(write_config):
    write_config('[toko]\napi_keys = "oops"\n')  # pragma: allowlist secret
    with pytest.raises(ValueError, match=r"Invalid api_keys .*expected a table"):
        load_config()


def test_non_string_api_key_is_rejected(write_config):
    write_config("[toko.api_keys]\nopenai = 123\n")
    with pytest.raises(ValueError, match=r"Invalid api_keys\.<redacted> in .*a string"):
        load_config()


def test_rejected_api_key_message_never_echoes_the_value(write_config):
    write_config(f'[toko.api_keys]\nanthropic = ["{SENTINEL}"]\n')
    with pytest.raises(ValueError, match=r"Invalid api_keys\.<redacted>") as excinfo:
        load_config()
    assert SENTINEL not in str(excinfo.value)


def test_rejected_api_key_message_never_echoes_the_key_name(write_config):
    """A key pasted into the name position is just as secret as one in the value."""
    write_config(f'[toko.api_keys]\n"{SENTINEL}" = 123\n')
    with pytest.raises(
        ValueError, match=r"Invalid api_keys\.<redacted> in .*a string"
    ) as excinfo:
        load_config()
    assert SENTINEL not in str(excinfo.value)


def test_rejected_api_key_message_never_echoes_a_dotted_key_name(write_config):
    """JWT-shaped keys carry dots, so redaction cannot split on the last one."""
    write_config(f'[toko.api_keys]\n"sk-ant.{SENTINEL}.XYZZY" = 123\n')
    with pytest.raises(
        ValueError, match=r"Invalid api_keys\.<redacted> in .*a string"
    ) as excinfo:
        load_config()
    assert SENTINEL not in str(excinfo.value)


def test_api_keys_written_as_a_bare_string_never_echo_the_value(write_config):
    """The likeliest misspelling puts the key itself where the table belongs."""
    write_config(f'[toko]\napi_keys = "{SENTINEL}"\n')
    with pytest.raises(
        ValueError, match=r"Invalid api_keys in .*expected a table"
    ) as excinfo:
        load_config()
    assert SENTINEL not in str(excinfo.value)


def test_string_exclude_patterns_is_rejected(write_config):
    write_config('[toko.exclude]\npatterns = "*.log"\n')
    with pytest.raises(
        ValueError, match=r"Invalid exclude\.patterns '\*\.log' in .*expected a list"
    ):
        load_config()


def test_non_string_exclude_pattern_is_rejected(write_config):
    write_config('[toko.exclude]\npatterns = ["*.log", 5]\n')
    with pytest.raises(
        ValueError, match=r"Invalid exclude\.patterns\[1\] 5 in .*expected a string"
    ):
        load_config()


def test_integer_booleans_are_accepted(write_config):
    """1/0 loaded and behaved correctly before these fields were validated."""
    write_config("[toko]\nrespect_gitignore = 0\nauto_update_prices = 1\n")

    config = load_config()
    assert config.respect_gitignore is False
    assert config.auto_update_prices is True


def test_other_integers_are_not_booleans(write_config):
    write_config("[toko]\nrespect_gitignore = 42\n")
    with pytest.raises(
        ValueError, match=r"Invalid respect_gitignore 42 in .*expected a boolean"
    ):
        load_config()
