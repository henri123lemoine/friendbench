from pathlib import Path

import yaml
from inspect_ai.model import GenerateConfig, get_model


def load_models(models_yaml: Path) -> list[dict]:
    return yaml.safe_load(models_yaml.read_text())


def model_configs(models_yaml: Path) -> list[dict]:
    results = []
    for m in load_models(models_yaml):
        if m.get("hidden"):
            continue
        model_id = m.get("id")
        if not model_id:
            continue
        gen_config = m.get("generation_config", {})
        results.append(
            {
                "id": model_id,
                "name": m["name"],
                "generation_config": gen_config,
            }
        )
    return results


def resolve_models(models_yaml: Path) -> list[dict]:
    results = []
    for entry in model_configs(models_yaml):
        gen_config = entry["generation_config"]
        config = GenerateConfig(**gen_config) if gen_config else GenerateConfig()
        model = get_model(entry["id"], config=config)
        results.append(
            {
                "id": entry["id"],
                "name": entry["name"],
                "model": model,
            }
        )
    return results
