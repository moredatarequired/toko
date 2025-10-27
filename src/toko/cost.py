"""Cost estimation using genai-prices."""

import contextlib

from genai_prices import Usage, calc_price

from toko.models import get_model


def estimate_cost(
    token_count: int,
    model: str,
    *,
    output_tokens: int = 0,
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
    model_info = get_model(model)
    usage = Usage(input_tokens=token_count, output_tokens=output_tokens)

    # Map provider names to genai-prices provider IDs
    provider_map = {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "google",
        "xai": "xai",
    }

    # Try direct provider first
    provider_id = provider_map.get(model_info.provider)
    if provider_id:
        with contextlib.suppress(Exception):
            price_data = calc_price(
                usage, model_ref=model_info.name, provider_id=provider_id
            )
            return float(price_data.total_price)

    # Try OpenRouter as fallback for all models
    # OpenRouter supports 548 models across many providers
    with contextlib.suppress(Exception):
        # OpenRouter uses different model name format
        # Try to convert common patterns to OpenRouter format
        openrouter_name = _convert_to_openrouter_name(
            model_info.name, model_info.provider
        )
        if openrouter_name:
            price_data = calc_price(
                usage, model_ref=openrouter_name, provider_id="openrouter"
            )
            return float(price_data.total_price)

    # No pricing available
    return None


def _convert_to_openrouter_name(model_name: str, provider: str) -> str | None:
    """Convert model name to OpenRouter format.

    Args:
        model_name: Original model name
        provider: Provider name

    Returns:
        OpenRouter-compatible model name, or None if no conversion available
    """
    # OpenRouter uses format like "provider/model-name"
    # Map common patterns to OpenRouter naming

    # Qwen models: Qwen/Qwen3-8B -> qwen/qwen-3-8b or similar
    if provider == "qwen":
        # Try common OpenRouter Qwen model names
        if "qwen3" in model_name.lower() or "qwen-3" in model_name.lower():
            return "qwen/qwen-2.5-72b-instruct"  # Map to available model
        if "qwen2.5" in model_name.lower() or "qwen-2.5" in model_name.lower():
            return "qwen/qwen-2.5-72b-instruct"
        if "qwen2" in model_name.lower() or "qwen-2" in model_name.lower():
            return "qwen/qwen-2-72b-instruct"
        return "qwen/qwen-2.5-72b-instruct"  # Default

    # DeepSeek models: deepseek-ai/DeepSeek-V3 -> deepseek/deepseek-chat
    if provider == "deepseek":
        if "v3" in model_name.lower() or "chat" in model_name.lower():
            return "deepseek/deepseek-chat"
        if "r1" in model_name.lower() or "reasoner" in model_name.lower():
            return "deepseek/deepseek-r1"
        return "deepseek/deepseek-chat"  # Default

    # Llama models: meta-llama/Llama-3.2-1B -> meta-llama/llama-3.2-...
    if provider == "llama":
        # OpenRouter uses lowercase and specific size variants
        if "3.3" in model_name:
            return "meta-llama/llama-3.3-70b-instruct"
        if "3.2" in model_name:
            if "90b" in model_name.lower():
                return "meta-llama/llama-3.2-90b-vision-instruct"
            if "11b" in model_name.lower():
                return "meta-llama/llama-3.2-11b-vision-instruct"
            if "3b" in model_name.lower():
                return "meta-llama/llama-3.2-3b-instruct"
            return "meta-llama/llama-3.2-1b-instruct"
        if "3.1" in model_name:
            if "405b" in model_name.lower():
                return "meta-llama/llama-3.1-405b-instruct"
            if "70b" in model_name.lower():
                return "meta-llama/llama-3.1-70b-instruct"
            return "meta-llama/llama-3.1-8b-instruct"
        if "3" in model_name:
            return "meta-llama/llama-3-8b-instruct"
        return "meta-llama/llama-3.2-1b-instruct"  # Default

    # Mistral models
    if provider == "mistral":
        if "large" in model_name.lower():
            return "mistralai/mistral-large"
        if "medium" in model_name.lower():
            return "mistralai/mistral-medium"
        if "small" in model_name.lower():
            return "mistralai/mistral-small"
        if "nemo" in model_name.lower():
            return "mistralai/mistral-nemo"
        if "7b" in model_name.lower():
            return "mistralai/mistral-7b-instruct"
        return "mistralai/mistral-small"  # Default

    # For models already in OpenRouter format or unknown providers
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
