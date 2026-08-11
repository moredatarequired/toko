"""Model registry and definitions."""

import json
import re
import sys
import tomllib
import typing
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources, util

from tiktoken.model import MODEL_TO_ENCODING as TIKTOKEN_MODEL_TO_ENCODING

from toko.config import get_models_path


@dataclass
class ModelInfo:
    """Information about a supported model."""

    name: str
    provider: str
    encoding: str | None = None  # For tiktoken models
    api_endpoint: str | None = None  # For API-based counting
    # ISO date the provider retired the model, or "unknown" when the provider
    # dropped it without publishing a date. Retired models stay in the registry
    # so toko can explain the failure instead of surfacing a raw 404.
    retired: str | None = None
    # Set when the provider still answers for a retired name but serves a
    # different model, which would otherwise mislabel the count.
    redirects_to: str | None = None
    # Tokenizer generation. Counts are only comparable within one generation, so
    # alias resolution must never map a name across two of these.
    tokenizer: str | None = None
    # Whether --list-models advertises the model.
    listed: bool = True


RETIREMENT_DATE_UNKNOWN = "unknown"

REGISTRY_FILENAME = "models.toml"

_STRING_FIELDS = (
    "provider",
    "encoding",
    "api_endpoint",
    "retired",
    "redirects_to",
    "tokenizer",
)

# Providers whose aliases are declared in the registry. Anthropic is absent on
# purpose: its aliases are derived under the Opus 4.7 tokenizer guard (see
# _build_anthropic_alias_map), and a declared alias could route a name to the
# other tokenizer generation, which is the one thing that must never happen.
_ALIASABLE_PROVIDERS = frozenset({"google", "xai"})

_PACKAGED_REGISTRY_SOURCE = "toko's packaged model registry"


@dataclass(frozen=True)
class Registry:
    models: dict[str, dict[str, ModelInfo]]
    aliases: dict[str, dict[str, str]]


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def _clean_entry(entry: object, source: str) -> tuple[str, dict[str, object]] | None:
    if not isinstance(entry, dict):
        _warn(f"{source}: ignoring a [[model]] entry that is not a table")
        return None
    table = typing.cast("dict[str, object]", entry)
    name = table.get("name")
    if not isinstance(name, str) or not name:
        _warn(f"{source}: ignoring a [[model]] entry with no 'name'")
        return None

    fields: dict[str, object] = {}
    for key, value in table.items():
        if key == "name":
            continue
        if key in _STRING_FIELDS:
            valid = isinstance(value, str)
        elif key == "listed":
            valid = isinstance(value, bool)
        elif key == "aliases":
            valid = isinstance(value, list) and all(
                isinstance(alias, str) for alias in value
            )
        else:
            _warn(f"{source}: ignoring unknown field '{key}' on {name}")
            continue
        if not valid:
            _warn(f"{source}: ignoring malformed field '{key}' on {name}")
            continue
        fields[key] = value
    return name, fields


def _merge_entries(
    documents: list[tuple[str, dict[str, object]]],
) -> dict[str, dict[str, object]]:
    """Merge registry documents field by field, later documents winning.

    A user entry that names an existing model updates only the fields it
    declares, so overriding one field cannot silently drop the others.
    """
    merged: dict[str, dict[str, object]] = {}
    for source, document in documents:
        entries = document.get("model", [])
        if not isinstance(entries, list):
            _warn(f"{source}: 'model' must be an array of tables")
            continue
        for entry in entries:
            cleaned = _clean_entry(entry, source)
            if cleaned is None:
                continue
            name, fields = cleaned
            merged.setdefault(name, {}).update(fields)
    return merged


def _string_field(fields: dict[str, object], key: str) -> str | None:
    value = fields.get(key)
    return value if isinstance(value, str) else None


def build_registry(documents: list[tuple[str, dict[str, object]]]) -> Registry:
    models: dict[str, dict[str, ModelInfo]] = defaultdict(dict)
    aliases: dict[str, dict[str, str]] = defaultdict(dict)

    for name, fields in _merge_entries(documents).items():
        provider = _string_field(fields, "provider")
        if not provider:
            _warn(f"ignoring model '{name}', which names no provider")
            continue
        listed = fields.get("listed")
        models[provider][name] = ModelInfo(
            # Google's CountTokens endpoint addresses models as "models/<name>".
            name=f"models/{name}" if provider == "google" else name,
            provider=provider,
            encoding=_string_field(fields, "encoding"),
            api_endpoint=_string_field(fields, "api_endpoint"),
            retired=_string_field(fields, "retired"),
            redirects_to=_string_field(fields, "redirects_to"),
            tokenizer=_string_field(fields, "tokenizer"),
            listed=listed if isinstance(listed, bool) else True,
        )

        declared = fields.get("aliases")
        if not isinstance(declared, list) or not declared:
            continue
        if provider not in _ALIASABLE_PROVIDERS:
            _warn(
                f"ignoring aliases on '{name}': {provider} models cannot declare them"
            )
            continue
        for alias in declared:
            if isinstance(alias, str):
                aliases[provider][alias] = name

    return Registry(models=dict(models), aliases=dict(aliases))


def _load_packaged_document() -> tuple[str, dict[str, object]]:
    resource = resources.files("toko.data").joinpath(REGISTRY_FILENAME)
    try:
        text = resource.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(
            f"{REGISTRY_FILENAME} is missing from the toko installation, so no "
            "models are known. Reinstall toko."
        ) from e
    try:
        return _PACKAGED_REGISTRY_SOURCE, tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise RuntimeError(f"{_PACKAGED_REGISTRY_SOURCE} is corrupt: {e}") from e


def _load_user_document() -> tuple[str, dict[str, object]] | None:
    """Read ~/.config/toko/models.toml, or nothing at all if it is unusable."""
    path = get_models_path()
    if not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            return str(path), tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        _warn(f"ignoring {path}: {e}")
        return None


@lru_cache
def load_registry() -> Registry:
    documents = [_load_packaged_document()]
    user_document = _load_user_document()
    if user_document is not None:
        documents.append(user_document)
    return build_registry(documents)


_REGISTRY = load_registry()

ANTHROPIC_MODELS = _REGISTRY.models.get("anthropic", {})
GOOGLE_MODELS = _REGISTRY.models.get("google", {})
XAI_MODELS = _REGISTRY.models.get("xai", {})
OPENAI_MODELS = _REGISTRY.models.get("openai", {})

_GOOGLE_ALIAS_MAP = _REGISTRY.aliases.get("google", {})
_XAI_ALIAS_MAP = _REGISTRY.aliases.get("xai", {})

# Anthropic replaced its tokenizer at Claude Opus 4.7: the same text produces
# roughly 30% more tokens on 4.7-generation models than on everything before
# them. Counting and cost therefore differ per generation, and the two must
# never be conflated -- see _build_anthropic_alias_map.
CLAUDE_TOKENIZER_OPUS_4_7 = "claude-opus-4-7"
CLAUDE_TOKENIZER_LEGACY = "claude-legacy"

# tiktoken cannot map dotted OpenAI names to a tokenizer at all, and its prefix
# table only grows on release. An encoding in the registry marks that name as
# verified, so counting stays exact and warning-free; anything absent is
# estimated with o200k_base and says so on stderr.
OPENAI_MODEL_ENCODINGS = {
    name: info.encoding for name, info in OPENAI_MODELS.items() if info.encoding
}

_OPENAI_NAME_PATTERN = re.compile(r"(gpt-|o\d)")


def detect_provider(model: str) -> str | None:
    """Detect provider from model name using pattern matching."""
    model_lower = model.lower()
    model_lower_base = model_lower.split("/")[-1]

    for tiktoken_model in TIKTOKEN_MODEL_TO_ENCODING:
        # If tiktoken prefix is in the model name, then the rest should be, e.g.
        # tiktoken includes gpt-5 which covers gpt-5, gpt-5-mini, gpt-5-nano, etc.
        if model_lower_base.startswith(tiktoken_model.lower()):
            return "openai"

    # Anthropic Claude models
    if "claude" in model_lower:
        return "anthropic"

    # Google Gemini/Gemma models
    if any(
        pattern in model_lower for pattern in ["gemini", "gemma"]
    ) and not model_lower.startswith("google/"):
        return "google"

    # xAI Grok models
    if "grok" in model_lower:
        return "xai"

    # Mistral models
    if "mistral" in model_lower or "mixtral" in model_lower:
        return "mistral"

    # Llama models (including fine-tunes)
    if "llama" in model_lower:
        return "llama"

    # DeepSeek models
    if "deepseek" in model_lower:
        return "deepseek"

    # Qwen models
    if "qwen" in model_lower:
        return "qwen"

    if model_lower.startswith("gpt-oss"):
        return "huggingface"

    if "/" in model_lower and not model_lower.startswith("models/"):
        return "huggingface"

    # Newer OpenAI names tiktoken has never heard of (gpt-6, gpt-5.6, o5). Checked
    # last so gpt-oss and org-prefixed names keep their more specific providers.
    if _OPENAI_NAME_PATTERN.match(model_lower):
        return "openai"

    return None


def _strip_anthropic_version(name: str) -> str:
    if len(name) > 9 and name[-9] == "-" and name[-8:].isdigit():
        return name[:-9]
    return name


def _build_anthropic_alias_map() -> dict[str, str]:
    candidates: dict[str, list[str]] = defaultdict(list)
    for canonical in ANTHROPIC_MODELS:
        alias = _strip_anthropic_version(canonical)
        if alias != canonical:
            candidates[alias].append(canonical)

    aliases: dict[str, str] = {}
    for alias, names in candidates.items():
        if len({ANTHROPIC_MODELS[name].tokenizer for name in names}) > 1:
            # The alias spans the Opus 4.7 tokenizer change, so resolving it
            # either way would report another generation's token count. Leave it
            # unresolvable and make the user name the model they mean.
            continue
        aliases[alias] = max(names)
    return aliases


_ANTHROPIC_ALIAS_MAP = _build_anthropic_alias_map()


def _resolve_anthropic_model(name: str) -> ModelInfo | None:
    if name in ANTHROPIC_MODELS:
        return ANTHROPIC_MODELS[name]

    normalized = name.removesuffix("-latest")
    if normalized in ANTHROPIC_MODELS:
        return ANTHROPIC_MODELS[normalized]

    canonical = _ANTHROPIC_ALIAS_MAP.get(_strip_anthropic_version(normalized))
    if canonical:
        return ANTHROPIC_MODELS[canonical]
    return None


def _normalize_google_model_name(name: str) -> str:
    if name.startswith("models/"):
        name = name.split("/", 1)[1]
    lowered = name.lower()
    if lowered in GOOGLE_MODELS:
        return lowered
    alias = _GOOGLE_ALIAS_MAP.get(lowered)
    if alias:
        return alias
    # Longest alias prefix wins. Matching in registry order would let the short
    # gemini-2.0-flash swallow variants of gemini-2.0-flash-lite, which is a
    # different model at a different price, and would make resolution depend on
    # where a user happened to add an alias in ~/.config/toko/models.toml.
    prefixes = [prefix for prefix in _GOOGLE_ALIAS_MAP if lowered.startswith(prefix)]
    if prefixes:
        return _GOOGLE_ALIAS_MAP[max(prefixes, key=len)]
    return lowered


def _resolve_google_model(name: str) -> ModelInfo | None:
    normalized = _normalize_google_model_name(name)
    return GOOGLE_MODELS.get(normalized)


def _resolve_xai_model(name: str) -> ModelInfo | None:
    lowered = name.lower()
    if lowered in XAI_MODELS:
        return XAI_MODELS[lowered]
    alias = _XAI_ALIAS_MAP.get(lowered)
    if alias:
        return XAI_MODELS.get(alias)
    return None


# All supported models
# Tokenizer aliases - map common shorthand to actual HuggingFace model paths
# These models share the same tokenizer within their family
TOKENIZER_ALIASES = {
    # Qwen family - all use same tokenizer
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2": "Qwen/Qwen2-7B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    # DeepSeek family
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "deepseek": "deepseek-ai/DeepSeek-V3",
    # Llama family
    "llama-3.1": "NousResearch/Hermes-3-Llama-3.1-8B",
    "llama": "NousResearch/Hermes-3-Llama-3.1-8B",
    # Misc popular open models
    "glm": "THUDM/glm-4-9b-chat",
    "phi": "microsoft/Phi-3.5-mini-instruct",
}

TRANSFORMERS_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3.5-vision-instruct",
    "THUDM/glm-4-9b-chat",
    "NousResearch/Hermes-3-Llama-3.1-8B",
)

MODELS = {**ANTHROPIC_MODELS, **GOOGLE_MODELS, **XAI_MODELS, **OPENAI_MODELS}


def _has_module(module: str) -> bool:
    return util.find_spec(module) is not None


@lru_cache
def _load_openrouter_entries() -> tuple[dict[str, object], ...]:
    try:
        resource = resources.files("toko.data").joinpath("openrouter_models.json")
    except FileNotFoundError:
        return ()
    try:
        payload = json.loads(resource.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return ()
    return tuple(entry for entry in payload if isinstance(entry, dict))


@lru_cache
def _openrouter_id_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in _load_openrouter_entries():
        hugging_face_id = entry.get("hugging_face_id")
        openrouter_id = entry.get("openrouter_id")
        if (
            isinstance(hugging_face_id, str)
            and hugging_face_id
            and isinstance(openrouter_id, str)
            and openrouter_id
        ):
            key = hugging_face_id.lower()
            existing = mapping.get(key)
            if existing and ":" not in existing and ":" in openrouter_id:
                continue
            mapping[key] = openrouter_id
    return mapping


def get_openrouter_id(model_name: str) -> str | None:
    return _openrouter_id_map().get(model_name.lower())


@dataclass(frozen=True)
class OptionalGroupDef:
    extra: str
    module: str
    providers: tuple[str, ...]
    models: tuple[str, ...]


OPTIONAL_GROUPS: tuple[OptionalGroupDef, ...] = (
    OptionalGroupDef(
        extra="mistral",
        module="mistral_common",
        providers=("mistral",),
        models=(
            "mistral-small-latest",
            "mistral-medium-latest",
            "mistral-large-latest",
        ),
    ),
    OptionalGroupDef(
        extra="transformers",
        module="transformers",
        providers=("llama", "deepseek", "qwen", "huggingface"),
        models=TRANSFORMERS_MODELS[:3],
    ),
)


def _make_basic_builder(provider: str) -> typing.Callable[[str], ModelInfo]:
    def builder(name: str) -> ModelInfo:
        return ModelInfo(name=name, provider=provider)

    return builder


def _build_openai_model(name: str) -> ModelInfo:
    return ModelInfo(
        name=name, provider="openai", encoding=OPENAI_MODEL_ENCODINGS.get(name.lower())
    )


def _build_google_model(name: str) -> ModelInfo:
    if name.startswith("models/"):
        model_name = name
    elif "/" in name:
        model_name = f"models/{name.split('/', 1)[1]}"
    else:
        model_name = f"models/{name}"
    return ModelInfo(name=model_name, provider="google")


_PROVIDER_BUILDERS: dict[str, typing.Callable[[str], ModelInfo]] = {
    "anthropic": _make_basic_builder("anthropic"),
    "google": _build_google_model,
    "openai": _build_openai_model,
    "xai": _make_basic_builder("xai"),
    "mistral": _make_basic_builder("mistral"),
    "llama": _make_basic_builder("llama"),
    "deepseek": _make_basic_builder("deepseek"),
    "qwen": _make_basic_builder("qwen"),
    "huggingface": _make_basic_builder("huggingface"),
}


def get_model(name: str) -> ModelInfo:
    """Get model info by name.

    Tries to find model in registry first, then checks tokenizer aliases,
    then falls back to dynamic detection.

    Args:
        name: Model name (can be full path or shorthand alias)

    Returns:
        ModelInfo for the model

    Raises:
        ValueError: If model provider cannot be detected
    """
    # First try the registry
    if name in MODELS:
        return MODELS[name]

    resolved_anthropic = _resolve_anthropic_model(name)
    if resolved_anthropic is not None:
        return resolved_anthropic

    resolved_google = _resolve_google_model(name)
    if resolved_google is not None:
        return resolved_google

    resolved_xai = _resolve_xai_model(name)
    if resolved_xai is not None:
        return resolved_xai

    # Check if it's a tokenizer alias (shorthand name)
    if name.lower() in TOKENIZER_ALIASES:
        canonical_name = TOKENIZER_ALIASES[name.lower()]
        provider = detect_provider(canonical_name) or "openai"
        return ModelInfo(name=canonical_name, provider=provider)

    # Fall back to dynamic detection
    provider = detect_provider(name)

    if provider is None:
        raise ValueError(
            f"Could not detect provider for model: {name}. "
            "Use --list-models to see known models, or ensure the model name "
            "contains a recognizable provider pattern (claude, gpt, gemini, etc.)"
        )

    builder = _PROVIDER_BUILDERS.get(provider)
    if builder is None:
        raise ValueError(
            f"Provider '{provider}' not supported. "
            "Supported providers: OpenAI, Anthropic, Google, xAI, Mistral, Llama, DeepSeek, Qwen. "
            f"Model name: {name}"
        )

    return builder(name)


def retirement_notice(model_info: ModelInfo) -> str | None:
    """Explain that a model is retired, and what the provider serves instead."""
    if model_info.retired is None:
        return None
    when = (
        "on an unpublished date"
        if model_info.retired == RETIREMENT_DATE_UNKNOWN
        else f"on {model_info.retired}"
    )
    notice = f"Warning: {model_info.name} was retired {when}"
    if model_info.redirects_to:
        return (
            f"{notice}; {model_info.provider} still answers for it but serves "
            f"{model_info.redirects_to}, so this count is {model_info.redirects_to}'s, "
            f"not {model_info.name}'s."
        )
    return f"{notice}; the {model_info.provider} API will reject or redirect it."


def warn_if_retired(model_info: ModelInfo) -> None:
    notice = retirement_notice(model_info)
    if notice is not None:
        print(notice, file=sys.stderr)


def list_models(*, include_retired: bool = False) -> dict[str, list[str]]:
    """List all supported models grouped by provider.

    Args:
        include_retired: Also list models the provider has retired. They stay in
            the registry so toko can explain the failure, but they are hidden by
            default because they can no longer be counted.

    Returns:
        Dictionary mapping provider name to list of model names
    """
    providers: dict[str, set[str]] = defaultdict(set)

    for model in MODELS.values():
        if not model.listed:
            continue
        if model.retired is not None and not include_retired:
            continue
        providers[model.provider].add(model.name)

    for model_name in TIKTOKEN_MODEL_TO_ENCODING:
        provider = detect_provider(model_name)
        if provider is None:
            provider = "openai"
        providers[provider].add(model_name)

    if _has_module("transformers"):
        for model_name in TRANSFORMERS_MODELS:
            provider = detect_provider(model_name) or "huggingface"
            providers[provider].add(model_name)

    return {
        provider: sorted(models, key=str.lower)
        for provider, models in providers.items()
    }


def list_optional_model_groups() -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for group in OPTIONAL_GROUPS:
        installed = _has_module(group.module)
        groups.append(
            {
                "extra": group.extra,
                "providers": list(group.providers),
                "models": list(group.models),
                "installed": installed,
            }
        )
    return groups
