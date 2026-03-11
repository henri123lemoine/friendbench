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

    prices = _load_prices()
    grader_model = None

    def _new_row():
        return {"input": 0, "output": 0, "cost": 0.0, "new_cost": 0.0, "has_unpriced": False}

    provider_rows = defaultdict(_new_row)
    grader_row = _new_row()
    rate_cache = {}

    fresh_usage = defaultdict(lambda: {"input": 0, "output": 0})
    for log in logs:
        if log.stats and log.stats.model_usage:
            for mid, u in log.stats.model_usage.items():
                fresh_usage[mid]["input"] += u.input_tokens
                fresh_usage[mid]["output"] += u.output_tokens

    for log in logs:
        if not log.samples:
            continue
        subject = log.eval.model
        for sample in log.samples:
            for ev in sample.events:
                if type(ev).__name__ != "ModelEvent":
                    continue
                if not (ev.output and ev.output.usage):
                    continue
                u = ev.output.usage
                model_id = ev.model
                is_grader = model_id != subject
                if is_grader:
                    grader_model = grader_model or model_id
                row = grader_row if is_grader else provider_rows[model_id.split("/")[0]]
                row["input"] += u.input_tokens
                row["output"] += u.output_tokens
                if model_id not in rate_cache:
                    rate_cache[model_id] = _lookup_price(prices, model_id)
                    if not rate_cache[model_id]:
                        row["has_unpriced"] = True
                rate = rate_cache[model_id]
                if rate:
                    row["cost"] += u.input_tokens * rate[0] + u.output_tokens * rate[1]

    for model_id, usage in fresh_usage.items():
        rate = rate_cache.get(model_id) or _lookup_price(prices, model_id)
        if not rate:
            continue
        is_grader = model_id == grader_model
        row = grader_row if is_grader else provider_rows[model_id.split("/")[0]]
        row["new_cost"] += usage["input"] * rate[0] + usage["output"] * rate[1]

    if not provider_rows:
        return

    def _fmt_tokens(n):
        return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"

    def _fmt(row):
        prefix = "~" if row["has_unpriced"] else " "
        cost_str = f"{prefix}${row['cost']:7.2f}" if row["cost"] or not row["has_unpriced"] else "      n/a"
        tokens = f"({_fmt_tokens(row['input'])} in, {_fmt_tokens(row['output'])} out)"
        if row["new_cost"] < row["cost"]:
            return f"{cost_str}  (${row['new_cost']:.2f} new)  {tokens}"
        return f"{cost_str}  {tokens}"

    all_rows = sorted(provider_rows.items(), key=lambda x: -x[1]["cost"])
    if grader_model:
        all_rows.append(("grader", grader_row))

    width = max(len(label) for label, _ in all_rows) + 2
    total_cost = sum(row["cost"] for _, row in all_rows)
    total_new = sum(row["new_cost"] for _, row in all_rows)
    any_unpriced = any(row["has_unpriced"] for _, row in all_rows)

    click.echo("\n  Cost estimate by provider:")
    for label, row in all_rows:
        suffix = f"  ({grader_model})" if label == "grader" else ""
        click.echo(f"    {label:<{width}} {_fmt(row)}{suffix}")
    prefix = "~" if any_unpriced else ""
    click.echo(f"    {'total':<{width}} {prefix}${total_cost:>6.2f}", nl=False)
    if total_new < total_cost:
        click.echo(f"  (${total_new:.2f} new)")
    else:
        click.echo()
    click.echo()


_LITELLM_PRICES_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm"
    "/main/model_prices_and_context_window.json"
)
_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_PROVIDER_REMAP = {"grok": "xai"}


def _load_prices():
    import json
    import time
    import urllib.request

    cache_dir = DEFAULT_INSPECT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_cached(url, filename, transform):
        path = cache_dir / filename
        try:
            if path.exists() and (time.time() - path.stat().st_mtime) / 3600 < 24:
                return json.loads(path.read_text())
            data = transform(json.loads(urllib.request.urlopen(url, timeout=5).read()))
            path.write_text(json.dumps(data))
            return data
        except Exception:
            return json.loads(path.read_text()) if path.exists() else {}

    litellm = _fetch_cached(_LITELLM_PRICES_URL, "model_prices.json", lambda d: d)
    openrouter = _fetch_cached(
        _OPENROUTER_MODELS_URL,
        "openrouter_prices.json",
        lambda d: {
            "openrouter/" + m["id"]: {
                "input_cost_per_token": float(m["pricing"]["prompt"]),
                "output_cost_per_token": float(m["pricing"]["completion"]),
            }
            for m in d["data"]
            if m.get("pricing")
        },
    )
    return {**litellm, **openrouter}


def _lookup_price(prices, model_id):
    if not prices:
        return None
    parts = model_id.split("/", 1)
    candidates = [model_id]
    if len(parts) == 2:
        provider, rest = parts
        candidates.append(rest)
        if provider in _PROVIDER_REMAP:
            candidates.append(_PROVIDER_REMAP[provider] + "/" + rest)
    for key in candidates:
        if key in prices:
            return _extract_rate(prices[key])
        if (match := _fuzzy_match(prices, key)):
            return _extract_rate(prices[match])
    return None


def _extract_rate(entry):
    return entry.get("input_cost_per_token", 0), entry.get("output_cost_per_token", 0)


import re

_DATE_SUFFIX_RE = re.compile(r".+-\d{4,}")


def _fuzzy_match(prices, name):
    bases = [name]
    if name.endswith("-0"):
        bases.append(name[:-2])
    for base in bases:
        for key in prices:
            if key.startswith(base + "-") and _DATE_SUFFIX_RE.fullmatch(key):
                return key
    return None


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

