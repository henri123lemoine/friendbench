import json
from collections import defaultdict
from pathlib import Path

import click

BENCHMARKS_DIR = Path(__file__).resolve().parent


def resolve_benchmark(name: str) -> Path:
    bench_dir = BENCHMARKS_DIR / name
    if not bench_dir.is_dir():
        raise click.ClickException(f"Unknown benchmark: {name}")
    return bench_dir


@click.group()
def bench():
    pass


@bench.group("eval")
def eval_group():
    pass


@eval_group.command()
@click.option("--benchmark", "-b", required=True, help="Benchmark name (e.g. friendbench, pressbench)")
@click.option("--models", multiple=True)
@click.option("--epochs", default=1, type=int)
@click.option("--log-dir", default="./logs")
@click.option("--category", multiple=True, help="Filter by category (friendbench)")
@click.option("--multi-turn", is_flag=True, help="Run only multi-turn pressure questions (friendbench)")
@click.option("--no-thinking", is_flag=True, help="Skip thinking variants")
@click.option("--thinking-only", is_flag=True, help="Run only thinking variants")
@click.option("--exclude", multiple=True, help="Exclude models whose name contains this string")
@click.option("--max-connections", default=None, type=int, help="Max concurrent API requests per model")
@click.option("--batch", is_flag=True, help="Use provider batch APIs (50% cheaper, ~24h turnaround)")
@click.option("--fail-on-error", default=None, type=float, help="Error threshold before failing (0-1 for proportion, >1 for count)")
@click.option("--no-fail-on-error", is_flag=True, help="Continue running even if samples error")
@click.option("--limit", default=None, type=str, help="Limit samples to evaluate (e.g. 10 or 10-20)")
@click.option("--max-retries", default=None, type=int, help="Max retries for model API requests")
@click.option("--cache", is_flag=False, flag_value="true", default=None, help="Cache model generations (optionally specify duration e.g. 7D)")
def run(
    benchmark, models, epochs, log_dir, category, multi_turn, no_thinking,
    thinking_only, exclude, max_connections, batch, fail_on_error,
    no_fail_on_error, limit, max_retries, cache,
):
    from inspect_ai import eval as inspect_eval
    from .models import model_entries

    bench_dir = resolve_benchmark(benchmark)
    tasks_file = bench_dir / "tasks.py"
    models_yaml = bench_dir / "data" / "models.yaml"

    task_args = {}
    if category:
        task_args["categories"] = ",".join(category)
    if multi_turn:
        task_args["multi_turn"] = True

    inspect_args = {k: v for k, v in {
        "max_connections": max_connections,
        "batch": batch or None,
        "fail_on_error": False if no_fail_on_error else fail_on_error,
        "limit": int(limit) if limit and limit.isdigit() else limit,
        "max_retries": max_retries,
        "cache": True if cache == "true" else cache,
    }.items() if v is not None}

    task_ref = f"{tasks_file}@{benchmark}"

    if models:
        logs = inspect_eval(
            task_ref,
            model=list(models),
            epochs=epochs,
            log_dir=log_dir,
            task_args=task_args,
            **inspect_args,
        )
    else:
        entries = model_entries(models_yaml)

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
            names = [e["name"] for e in group]
            click.echo(f"\n  Running: {', '.join(names)}\n")
            result = inspect_eval(
                task_ref,
                model=model_ids,
                epochs=epochs,
                log_dir=log_dir,
                task_args=task_args,
                **config,
                **inspect_args,
            )
            logs.extend(result)

    entries = model_entries(models_yaml) if not models else []
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
        if log.status == "success" and log.results:
            score = log.results.scores[0].metrics
            if "accuracy" in score:
                val = f"{score['accuracy'].value:.0%}"
            elif "mean" in score:
                val = f"{score['mean'].value:.1f}"
            else:
                val = str(next(iter(score.values())).value)
            rows.append((display, val))
        else:
            rows.append((display, log.status.upper()))

    if not rows:
        return
    rows.sort(key=lambda r: r[1], reverse=True)
    width = max(len(r[0]) for r in rows) + 2

    click.echo()
    for model, result in rows:
        click.echo(f"  {model:<{width}} {result}")


@eval_group.command("list-models")
@click.option("--benchmark", "-b", required=True, help="Benchmark name")
def list_models(benchmark):
    from .models import model_entries

    bench_dir = resolve_benchmark(benchmark)
    models_yaml = bench_dir / "data" / "models.yaml"

    for entry in model_entries(models_yaml):
        config = entry["generation_config"]
        config_str = f"  {config}" if config else ""
        click.echo(f"{entry['name']:30s} {entry['id']}{config_str}")
