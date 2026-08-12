"""Cost estimation using genai-prices."""

import contextlib

from genai_prices import Usage, calc_price

from toko.models import get_model, get_openrouter_id

PROVIDER_ID_MAP = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "xai": "xai",
}


def _calculate_price(usage: Usage, *, model_ref: str, provider_id: str) -> float | None:
    with contextlib.suppress(Exception):
        price_data = calc_price(usage, model_ref=model_ref, provider_id=provider_id)
        return float(price_data.total_price)
    return None


def estimate_cost(
    token_count: int, model: str, *, output_tokens: int = 0
) -> float | None:
    """Estimate cost for a given token count and model.

    Args:
        token_count: Number of input tokens
        model: Model name
        output_tokens: Number of output tokens (default 0)

    Returns:
        Total cost in USD, or None if price not available

    Note:
        Prices are estimates and may not be 100% accurate. See genai-prices
        documentation for more information. For models without direct provider
        pricing, falls back to OpenRouter pricing (548 models supported).
    """
    try:
        model_info = get_model(model)
    except ValueError:
        return None
    usage = Usage(input_tokens=token_count, output_tokens=output_tokens)

    # Map provider names to genai-prices provider IDs
    provider_id = PROVIDER_ID_MAP.get(model_info.provider)
    price = None
    if provider_id:
        # The name the user asked for comes first: alias resolution collapses
        # variants onto a canonical name that genai-prices either prices
        # differently or does not know at all (Google's "models/" prefix).
        for model_ref in dict.fromkeys((model, model_info.name)):
            price = _calculate_price(
                usage, model_ref=model_ref, provider_id=provider_id
            )
            if price is not None:
                return price

    openrouter_name = _convert_to_openrouter_name(model_info.name, model_info.provider)
    if openrouter_name is None:
        openrouter_name = get_openrouter_id(model_info.name)

    if openrouter_name:
        price = _calculate_price(
            usage, model_ref=openrouter_name, provider_id="openrouter"
        )
    return price


def _convert_qwen_name(model_name: str) -> str:
    lower = model_name.lower()
    if "qwen3" in lower or "qwen-3" in lower:
        return "qwen/qwen-2.5-72b-instruct"
    if "qwen2.5" in lower or "qwen-2.5" in lower:
        return "qwen/qwen-2.5-72b-instruct"
    if "qwen2" in lower or "qwen-2" in lower:
        return "qwen/qwen-2-72b-instruct"
    return "qwen/qwen-2.5-72b-instruct"


def _convert_deepseek_name(model_name: str) -> str:
    lower = model_name.lower()
    if "r1" in lower or "reasoner" in lower:
        return "deepseek/deepseek-r1"
    return "deepseek/deepseek-chat"


def _convert_llama_name(model_name: str) -> str:
    lower = model_name.lower()
    if "3.3" in model_name:
        return "meta-llama/llama-3.3-70b-instruct"
    if "3.2" in model_name:
        if "90b" in lower:
            return "meta-llama/llama-3.2-90b-vision-instruct"
        if "11b" in lower:
            return "meta-llama/llama-3.2-11b-vision-instruct"
        if "3b" in lower:
            return "meta-llama/llama-3.2-3b-instruct"
        return "meta-llama/llama-3.2-1b-instruct"
    if "3.1" in model_name:
        if "405b" in lower:
            return "meta-llama/llama-3.1-405b-instruct"
        if "70b" in lower:
            return "meta-llama/llama-3.1-70b-instruct"
        return "meta-llama/llama-3.1-8b-instruct"
    if "3" in model_name:
        return "meta-llama/llama-3-8b-instruct"
    return "meta-llama/llama-3.2-1b-instruct"


def _convert_mistral_name(model_name: str) -> str:
    lower = model_name.lower()
    # Left to the size-word branches below, each of these families lands on an unrelated
    # release: open-mixtral-8x7b matches "7b" and bills as mistral-7b-instruct, and any
    # name matching no size word at all falls through to the mistral-small default.
    if "pixtral" in lower:
        return (
            "mistralai/pixtral-large-2411"
            if "large" in lower
            else "mistralai/pixtral-12b"
        )
    if "codestral" in lower:
        if "mamba" in lower:
            return "mistralai/codestral-mamba"
        if "2501" in lower:
            return "mistralai/codestral-2501"
        # A rolling codestral name still has to resolve to some release, and the newest
        # one carried beats the mistral-small fallback. codestral-2405 and codestral-22b
        # fall through to that fallback on purpose: neither release is in the data, and
        # 2501/2508 are different models rather than later names for them.
        if "2405" not in lower and "22b" not in lower:
            return "mistralai/codestral-2508"
    if "ministral" in lower and "8b" in lower:
        return "mistralai/ministral-8b"
    if "ministral" in lower and "3b" in lower:
        return "mistralai/ministral-3b"
    if "mixtral" in lower:
        return (
            "mistralai/mixtral-8x22b-instruct"
            if "8x22b" in lower
            else "mistralai/mixtral-8x7b-instruct"
        )
    if "large" in lower:
        return "mistralai/mistral-large"
    if "medium" in lower:
        # The unversioned mistralai/mistral-medium is the retired 2312 release at nearly
        # seven times Medium 3's rate, so only 2312 itself may have it. Everything else
        # goes where the native Mistral data sends any mistral-medium* name: Medium 3.
        return (
            "mistralai/mistral-medium"
            if "2312" in lower
            else "mistralai/mistral-medium-3"
        )
    if "small" in lower:
        # Same shape one tier down, but only the rolling name moved on: the dated small
        # releases really are the unversioned entry, while mistral-small-latest is
        # Mistral Small 3.2 in the native data.
        return (
            "mistralai/mistral-small-3.2-24b-instruct"
            if "latest" in lower
            else "mistralai/mistral-small"
        )
    if "tiny" in lower:
        return "mistralai/mistral-tiny"
    if "nemo" in lower:
        return "mistralai/mistral-nemo"
    if "7b" in lower:
        return "mistralai/mistral-7b-instruct"
    return "mistralai/mistral-small"


_OPENROUTER_CONVERTERS = {
    "qwen": _convert_qwen_name,
    "deepseek": _convert_deepseek_name,
    "llama": _convert_llama_name,
    "mistral": _convert_mistral_name,
}


def _convert_to_openrouter_name(model_name: str, provider: str) -> str | None:
    converter = _OPENROUTER_CONVERTERS.get(provider)
    if converter:
        return converter(model_name)
    return None


def format_cost(cost: float | None) -> str:
    """Format cost for display.

    Args:
        cost: Cost in USD or None

    Returns:
        Formatted cost string
    """
    if cost is None:
        return "N/A"

    # Format with appropriate precision
    if cost < 0.0001:
        return f"${cost:.6f}"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"
