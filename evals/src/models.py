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


def model_entries() -> list[dict]:
    results = []
    for m in load_models():
        if m.get("hidden"):
            continue
        mid = inspect_model_id(m)
        if not mid:
            continue
        results.append({
            "id": mid,
            "name": m["name"],
            "generation_config": m.get("generation_config", {}),
        })
    return results


def default_models() -> list[str]:
    return [e["id"] for e in model_entries()]
