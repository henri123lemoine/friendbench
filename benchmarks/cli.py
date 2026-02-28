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
    from .models import resolve_models

    bench_dir = resolve_benchmark(benchmark)
    tasks_file = bench_dir / "tasks.py"
    models_yaml = bench_dir / "data" / "models.yaml"
    task_ref = f"{tasks_file}@{benchmark}"

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
        entries = resolve_models(models_yaml)

        if no_thinking:
            entries = [e for e in entries if not _is_thinking(e)]
        if thinking_only:
            entries = [e for e in entries if _is_thinking(e)]
        if exclude:
            entries = [
                e for e in entries
                if not any(x.lower() in e["name"].lower() for x in exclude)
            ]

        model_list = [e["model"] for e in entries]
        names = [e["name"] for e in entries]
        click.echo(f"\n  Running: {', '.join(names)}\n")

        logs = inspect_eval(
            task_ref,
            model=model_list,
            epochs=epochs,
            log_dir=log_dir,
            task_args=task_args,
            **inspect_args,
        )

    _print_results(logs, models_yaml if not models else None)


def _is_thinking(entry: dict) -> bool:
    config = entry["model"].config
    return bool(config.reasoning_effort) or bool(config.reasoning_tokens)


def _print_results(logs, models_yaml):
    from .models import resolve_models

    name_lookup = {}
    if models_yaml:
        for e in resolve_models(models_yaml):
            m = e["model"]
            key = (e["id"], m.config.model_dump_json(exclude_none=True))
            name_lookup[key] = e["name"]

    rows = []
    for log in logs:
        key = (log.eval.model, log.eval.model_generate_config.model_dump_json(exclude_none=True))
        display = name_lookup.get(key, log.eval.model)
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
    from .models import model_configs

    bench_dir = resolve_benchmark(benchmark)
    models_yaml = bench_dir / "data" / "models.yaml"

    for entry in model_configs(models_yaml):
        gen = entry["generation_config"]
        config_str = f"  ({gen})" if gen else ""
        click.echo(f"{entry['name']:30s} {entry['id']}{config_str}")
