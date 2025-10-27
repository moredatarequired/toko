"""Fetch OpenRouter model metadata and persist Hugging Face mappings."""

import json
from pathlib import Path
from typing import cast

import httpx

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OUTPUT_PATH = Path("src/toko/data/openrouter_models.json")

# Curated list of open-source models whose tokenizers are publicly available
# and can be loaded without additional dependencies beyond `transformers`.
SUPPORTED_HF_MODELS = {
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "THUDM/glm-4-9b-chat",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3.5-vision-instruct",
    "NousResearch/Hermes-3-Llama-3.1-8B",
}


def fetch_models() -> list[dict[str, object]]:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(OPENROUTER_MODELS_URL)
        response.raise_for_status()
        payload = response.json()

    raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        raise TypeError("Unexpected response payload: missing 'data' list")
    return raw_models


def extract_huggingface_models(
    raw_models: list[dict[str, object]],
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for item in raw_models:
        hugging_face_id = item.get("hugging_face_id")
        if not hugging_face_id:
            continue
        if not isinstance(hugging_face_id, str):
            continue
        if hugging_face_id not in SUPPORTED_HF_MODELS:
            continue
        entry = {"hugging_face_id": hugging_face_id}
        model_id = item.get("id")
        if isinstance(model_id, str):
            entry["openrouter_id"] = model_id
        architecture = item.get("architecture")
        if isinstance(architecture, dict):
            arch_dict = cast("dict[str, object]", architecture)
            tokenizer_value = arch_dict.get("tokenizer")
            if isinstance(tokenizer_value, str) and tokenizer_value:
                entry["tokenizer"] = tokenizer_value
        entries.append(entry)
    return entries


def main() -> int:
    raw_models = fetch_models()
    entries = extract_huggingface_models(raw_models)
    entries.sort(key=lambda item: item["hugging_face_id"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(entries)} Hugging Face mappings to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
