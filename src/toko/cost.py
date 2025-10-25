"""Cost estimation using genai-prices."""

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
        documentation for more information.
    """
    model_info = get_model(model)

    # Map provider names to genai-prices provider IDs
    provider_map = {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "google",
        "xai": "xai",
    }

    provider_id = provider_map.get(model_info.provider)
    if not provider_id:
        return None

    try:
        usage = Usage(input_tokens=token_count, output_tokens=output_tokens)
        price_data = calc_price(
            usage, model_ref=model_info.name, provider_id=provider_id
        )
        # Convert Decimal to float
        return float(price_data.total_price)
    except Exception:
        # Price data not available for this model
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
