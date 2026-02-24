from pathlib import Path

import yaml

PROVIDER_PREFIXES = {
    "Anthropic": "anthropic",
    "OpenAI": "openai",
    "xAI": "grok",
    "DeepSeek": "deepseek",
    "Google": "google",
}

MODELS_YAML = Path(__file__).resolve().parents[2] / "data" / "models.yaml"


def load_models() -> list[dict]:
    return yaml.safe_load(MODELS_YAML.read_text())


def inspect_model_id(entry: dict) -> str | None:
    api_id = entry.get("api_id")
    if not api_id:
        return None
    prefix = PROVIDER_PREFIXES[entry["provider"]]
    return f"{prefix}/{api_id}"


def default_models() -> list[str]:
    return [
        mid
        for m in load_models()
        if not m.get("hidden")
        for mid in [inspect_model_id(m)]
        if mid
    ]
