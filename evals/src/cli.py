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
def run(models, epochs, log_dir, category, multi_turn):
    from inspect_ai import eval as inspect_eval
    from .models import default_models

    models = list(models) or default_models()

    task_args = {}
    if category:
        task_args["categories"] = ",".join(category)
    if multi_turn:
        task_args["multi_turn"] = True

    logs = inspect_eval(
        f"{TASKS_FILE}@friendbench",
        model=models,
        epochs=epochs,
        log_dir=log_dir,
        task_args=task_args,
    )

    rows = []
    for log in logs:
        model = log.eval.model
        if log.status == "success":
            acc = log.results.scores[0].metrics["accuracy"].value
            rows.append((model, f"{acc:.0%}"))
        else:
            rows.append((model, log.status.upper()))

    rows.sort(key=lambda r: r[1], reverse=True)
    width = max(len(r[0]) for r in rows) + 2

    click.echo()
    for model, result in rows:
        click.echo(f"  {model:<{width}} {result}")


@eval_group.command("list-models")
def list_models():
    from .models import load_models, inspect_model_id

    for entry in load_models():
        mid = inspect_model_id(entry)
        if mid and not entry.get("hidden"):
            click.echo(mid)
