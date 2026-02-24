import importlib.util
import sys
from pathlib import Path

import click

TASKS_FILE = Path(__file__).parent / "tasks.py"

SCORED_TASKS = [
    f"{TASKS_FILE}@mc_friendliness",
    f"{TASKS_FILE}@mc_sycophancy",
    f"{TASKS_FILE}@pattern_honesty",
    f"{TASKS_FILE}@rubric_warmth",
    f"{TASKS_FILE}@rubric_pushback",
]

VIBES_TASKS = [
    f"{TASKS_FILE}@human_vibes",
]


def _task_has_samples(task_ref: str) -> bool:
    file_path, func_name = task_ref.rsplit("@", 1)
    mod_name = f"_task_check_{Path(file_path).stem}"
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    mod = sys.modules[mod_name]
    try:
        getattr(mod, func_name)()
        return True
    except ValueError:
        return False


@click.group()
def fb():
    pass


@fb.group("eval")
def eval_group():
    pass


@eval_group.command()
@click.option("--models", multiple=True, required=True)
@click.option("--tasks", multiple=True)
@click.option("--include-vibes", is_flag=True)
@click.option("--log-dir", default="./logs")
def run(models, tasks, include_vibes, log_dir):
    from inspect_ai import eval_set

    models = list(models)
    tasks = list(tasks) or list(SCORED_TASKS)
    if include_vibes:
        tasks = tasks + VIBES_TASKS

    valid_tasks = []
    for t in tasks:
        if _task_has_samples(t):
            valid_tasks.append(t)
        else:
            name = t.rsplit("@", 1)[-1]
            click.echo(f"Warning: skipping '{name}' (no samples)", err=True)

    if not valid_tasks:
        click.echo("Error: all tasks have empty datasets. Add samples before running.", err=True)
        raise SystemExit(1)

    try:
        eval_set(valid_tasks, model=models, log_dir=log_dir)
    except Exception as e:
        if "key" in str(e).lower() or "auth" in str(e).lower() or "credential" in str(e).lower():
            click.echo(f"Error: API authentication failed — {e}", err=True)
            raise SystemExit(1)
        raise


@eval_group.command("list-models")
def list_models():
    from .models import load_models, inspect_model_id

    for entry in load_models():
        mid = inspect_model_id(entry)
        if mid and not entry.get("hidden"):
            click.echo(mid)


@eval_group.command()
@click.option("--log-dir", default="./logs")
def score(log_dir):
    if not Path(log_dir).exists():
        click.echo("No logs directory found. Run some evals first.")
        return

    from inspect_ai.analysis import evals_df, EvalInfo, EvalModel, EvalScores

    df = evals_df(log_dir, columns=EvalInfo + EvalModel + EvalScores)
    df = df[df["status"] == "success"]

    if df.empty:
        click.echo("No scored logs found.")
        return

    accuracy_cols = [c for c in df.columns if c.startswith("score_") and "accuracy" in c]
    if not accuracy_cols:
        click.echo("No accuracy scores found.")
        return

    df["mean_accuracy"] = df[accuracy_cols].mean(axis=1)
    summary = df.groupby("model")["mean_accuracy"].mean() * 100

    click.echo(f"\n{'Model':<45} {'Proxy Score':>12}")
    click.echo("-" * 58)
    for model, proxy_score in summary.sort_values(ascending=False).items():
        click.echo(f"{model:<45} {proxy_score:>10.1f}/100")
