import os
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

BENCHMARKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS_DIR.parent
DEFAULT_INSPECT_CACHE_DIR = REPO_ROOT / ".inspect-cache"
os.environ.setdefault("INSPECT_CACHE_DIR", str(DEFAULT_INSPECT_CACHE_DIR))
TEST_MODELS = ["anthropic/claude-haiku-4-5-20251001", "openai/gpt-4o-mini"]


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
@click.option(
    "--benchmark",
    "-b",
    required=True,
    help="Benchmark name (e.g. friendbench, pressbench)",
)
@click.option("--models", multiple=True)
@click.option("--epochs", default=1, type=int)
@click.option("--log-dir", default="./logs")
@click.option("--category", multiple=True, help="Filter by category (friendbench)")
@click.option("--no-thinking", is_flag=True, help="Skip thinking variants")
@click.option("--thinking-only", is_flag=True, help="Run only thinking variants")
@click.option(
    "--exclude", multiple=True, help="Exclude models whose name contains this string"
)
@click.option(
    "--max-connections",
    default=None,
    type=int,
    help="Max concurrent API requests per model",
)
@click.option(
    "--batch",
    is_flag=True,
    help="Use provider batch APIs (50% cheaper, ~24h turnaround)",
)
@click.option(
    "--fail-on-error",
    default=None,
    type=float,
    help="Error threshold before failing (0-1 for proportion, >1 for count)",
)
@click.option(
    "--no-fail-on-error", is_flag=True, help="Continue running even if samples error"
)
@click.option(
    "--limit",
    default=None,
    type=str,
    help="Limit samples to evaluate (e.g. 10 or 10-20)",
)
@click.option(
    "--max-retries", default=None, type=int, help="Max retries for model API requests"
)
@click.option("--no-cache", is_flag=True, help="Disable caching of model generations")
@click.option(
    "--test",
    is_flag=True,
    help="Smoke-test with cheap models and a representative sample subset",
)
def run(
    benchmark,
    models,
    epochs,
    log_dir,
    category,
    no_thinking,
    thinking_only,
    exclude,
    max_connections,
    batch,
    fail_on_error,
    no_fail_on_error,
    limit,
    max_retries,
    no_cache,
    test,
):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import CachePolicy
    from .models import resolve_models

    bench_dir = resolve_benchmark(benchmark)
    tasks_file = bench_dir / "tasks.py"
    models_yaml = bench_dir / "data" / "models.yaml"
    task_ref = f"{tasks_file}@{benchmark}"

    task_args = {}
    if category:
        task_args["categories"] = ",".join(category)
    if test:
        task_args["test"] = True

    inspect_args = {
        k: v
        for k, v in {
            "max_connections": max_connections,
            "batch": batch or None,
            "fail_on_error": False if no_fail_on_error else fail_on_error,
            "limit": int(limit) if limit and limit.isdigit() else limit,
            "max_retries": max_retries,
            "cache": CachePolicy(expiry=None) if not no_cache else None,
        }.items()
        if v is not None
    }

    if test:
        click.echo(f"\n  [test] Running: {', '.join(TEST_MODELS)}\n")
        logs = inspect_eval(
            task_ref,
            model=TEST_MODELS,
            epochs=1,
            log_dir=log_dir,
            task_args=task_args,
            **inspect_args,
        )
    elif models:
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
                e
                for e in entries
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

    name_lookup = _model_name_lookup(models_yaml)
    _print_results(logs, name_lookup)
    _print_costs(logs)

    if not test:
        _save_scores(logs, bench_dir, name_lookup)


def _is_thinking(entry: dict) -> bool:
    if "thinking" in entry:
        return bool(entry["thinking"])

    config = entry["model"].config
    return (
        config.reasoning_effort not in (None, "none")
        or bool(config.reasoning_tokens)
    )


def _model_name_lookup(models_yaml):
    from inspect_ai.model import GenerateConfig

    from .models import model_configs

    lookup = {}
    for entry in model_configs(models_yaml):
        gen = entry["generation_config"]
        config = GenerateConfig(**gen) if gen else GenerateConfig()
        key = (entry["id"], config.model_dump_json(exclude_none=True))
        lookup[key] = entry["name"]
    return lookup


def _log_score(log):
    metrics = log.results.scores[0].metrics
    if "accuracy" in metrics:
        return metrics["accuracy"].value, True
    if "mean" in metrics:
        return metrics["mean"].value, False
    return next(iter(metrics.values())).value, False


def _log_name(log, name_lookup):
    key = (
        log.eval.model,
        log.eval.model_generate_config.model_dump_json(exclude_none=True),
    )
    return name_lookup.get(key, log.eval.model)


def _print_results(logs, name_lookup):
    rows = []
    for log in logs:
        name = _log_name(log, name_lookup)
        if log.status == "success" and log.results:
            val, is_pct = _log_score(log)
            rows.append((name, f"{val:.0%}" if is_pct else f"{val:.1f}"))
        else:
            rows.append((name, log.status.upper()))

    if not rows:
        return
    rows.sort(key=lambda r: r[1], reverse=True)
    width = max(len(r[0]) for r in rows) + 2

    click.echo()
    for model, result in rows:
        click.echo(f"  {model:<{width}} {result}")


def _print_costs(logs):
    from collections import defaultdict

    fields = ("input", "output", "cache_read", "cache_write", "total_cost")
    provider_costs = defaultdict(lambda: dict.fromkeys(fields, 0.0))

    for log in logs:
        if not log.stats or not log.stats.model_usage:
            continue
        for model_id, usage in log.stats.model_usage.items():
            provider = model_id.split("/")[0]
            p = provider_costs[provider]
            p["input"] += usage.input_tokens
            p["output"] += usage.output_tokens
            p["cache_read"] += usage.input_tokens_cache_read or 0
            p["cache_write"] += usage.input_tokens_cache_write or 0
            if usage.total_cost is not None:
                p["total_cost"] += usage.total_cost

    if not provider_costs:
        return

    click.echo("\n  Cost breakdown by provider:")
    total = 0.0
    width = max(len(p) for p in provider_costs) + 2
    for provider, d in sorted(provider_costs.items(), key=lambda x: -x[1]["total_cost"]):
        total += d["total_cost"]
        cost_str = f"${d['total_cost']:.2f}" if d["total_cost"] else "n/a"
        inp = d["input"] / 1_000_000
        out = d["output"] / 1_000_000
        parts = [f"{inp:.1f}M in", f"{out:.1f}M out"]
        if d["cache_read"]:
            parts.append(f"{d['cache_read'] / 1_000_000:.1f}M cached")
        click.echo(f"    {provider:<{width}} {cost_str:>8}  ({', '.join(parts)})")
    click.echo(f"    {'total':<{width}} {'$' + f'{total:.2f}':>8}")
    click.echo()


def _save_scores(logs, bench_dir, name_lookup):
    import subprocess

    import yaml

    from .models import load_scores

    data_dir = bench_dir / "data"
    updates = {}
    for log in logs:
        if log.status != "success" or not log.results:
            continue
        name = _log_name(log, name_lookup)
        val, is_pct = _log_score(log)
        updates[name] = round(val * 100) if is_pct else round(val, 1)

    if not updates:
        return

    scores = dict(sorted({**load_scores(data_dir), **updates}.items()))
    (data_dir / "scores.yaml").write_text(
        yaml.dump(scores, default_flow_style=False, allow_unicode=True)
    )

    result = subprocess.run(
        ["node", str(REPO_ROOT / "netlify" / "build.js"), bench_dir.name],
        capture_output=True,
    )
    if result.returncode != 0:
        click.echo(f"\n  Warning: build.js failed: {result.stderr.decode().strip()}")

    click.echo(
        f"\n  Scores saved: {', '.join(f'{n} ({v})' for n, v in sorted(updates.items()))}"
    )


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

