import json
from collections import defaultdict
from pathlib import Path

import click

TASKS_FILE = Path(__file__).parent / "tasks.py"


@click.group()
def fb():
    pass


@fb.group("eval")
def eval_group():
    pass


@eval_group.command()
@click.option("--models", multiple=True)
@click.option("--epochs", default=1, type=int)
@click.option("--log-dir", default="./logs")
@click.option("--category", multiple=True, help="Filter by category (e.g. sycophancy, relationship)")
@click.option("--multi-turn", is_flag=True, help="Run only multi-turn pressure questions")
@click.option("--no-thinking", is_flag=True, help="Skip thinking variants")
@click.option("--thinking-only", is_flag=True, help="Run only thinking variants")
@click.option("--exclude", multiple=True, help="Exclude models whose name contains this string")
def run(models, epochs, log_dir, category, multi_turn, no_thinking, thinking_only, exclude):
    from inspect_ai import eval as inspect_eval
    from .models import model_entries

    task_args = {}
    if category:
        task_args["categories"] = ",".join(category)
    if multi_turn:
        task_args["multi_turn"] = True

    if models:
        logs = inspect_eval(
            f"{TASKS_FILE}@friendbench",
            model=list(models),
            epochs=epochs,
            log_dir=log_dir,
            task_args=task_args,
        )
    else:
        entries = model_entries()

        if no_thinking:
            entries = [e for e in entries if not e["generation_config"]]
        if thinking_only:
            entries = [e for e in entries if e["generation_config"]]
        if exclude:
            entries = [
                e for e in entries
                if not any(x.lower() in e["name"].lower() for x in exclude)
            ]

        groups = defaultdict(list)
        for e in entries:
            key = json.dumps(e["generation_config"], sort_keys=True)
            groups[key].append(e)

        logs = []
        for config_json, group in groups.items():
            config = json.loads(config_json)
            model_ids = [e["id"] for e in group]
            result = inspect_eval(
                f"{TASKS_FILE}@friendbench",
                model=model_ids,
                epochs=epochs,
                log_dir=log_dir,
                task_args=task_args,
                **config,
            )
            logs.extend(result)

    entries = model_entries() if not models else []
    name_lookup = {}
    for e in entries:
        gc = e["generation_config"]
        has_thinking = (
            bool(gc.get("reasoning_effort") and gc["reasoning_effort"] != "none")
            or bool(gc.get("reasoning_tokens"))
        )
        name_lookup[(e["id"], has_thinking)] = e["name"]

    rows = []
    for log in logs:
        model_id = log.eval.model
        config = log.eval.config
        thinking = (
            getattr(config, "reasoning_effort", None) not in (None, "none")
            or getattr(config, "reasoning_tokens", None) not in (None, 0)
        )
        display = name_lookup.get((model_id, thinking), model_id)
        if log.status == "success":
            acc = log.results.scores[0].metrics["accuracy"].value
            rows.append((display, f"{acc:.0%}"))
        else:
            rows.append((display, log.status.upper()))

    rows.sort(key=lambda r: r[1], reverse=True)
    width = max(len(r[0]) for r in rows) + 2

    click.echo()
    for model, result in rows:
        click.echo(f"  {model:<{width}} {result}")


@eval_group.command("list-models")
def list_models():
    from .models import model_entries

    for entry in model_entries():
        config = entry["generation_config"]
        config_str = f"  {config}" if config else ""
        click.echo(f"{entry['name']:30s} {entry['id']}{config_str}")
