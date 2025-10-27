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
        price = _calculate_price(
            usage, model_ref=model_info.name, provider_id=provider_id
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
    if "large" in lower:
        return "mistralai/mistral-large"
    if "medium" in lower:
        return "mistralai/mistral-medium"
    if "small" in lower:
        return "mistralai/mistral-small"
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
