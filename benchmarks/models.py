from pathlib import Path

import yaml

PROVIDER_PREFIXES = {
    "Anthropic": "anthropic",
    "OpenAI": "openai",
    "xAI": "grok",
    "DeepSeek": "deepseek",
    "Google": "google",
}


def load_models(models_yaml: Path) -> list[dict]:
    return yaml.safe_load(models_yaml.read_text())


def inspect_model_id(entry: dict) -> str | None:
    api_id = entry.get("api_id")
    if not api_id:
        return None
    prefix = PROVIDER_PREFIXES[entry["provider"]]
    return f"{prefix}/{api_id}"


def model_entries(models_yaml: Path) -> list[dict]:
    results = []
    for m in load_models(models_yaml):
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
