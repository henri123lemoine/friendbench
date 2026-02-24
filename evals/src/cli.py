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
@click.option("--models", multiple=True, required=True)
@click.option("--log-dir", default="./logs")
def run(models, log_dir):
    from inspect_ai import eval as inspect_eval

    inspect_eval(
        f"{TASKS_FILE}@friendbench",
        model=list(models),
        log_dir=log_dir,
    )


@eval_group.command("list-models")
def list_models():
    from .models import load_models, inspect_model_id

    for entry in load_models():
        mid = inspect_model_id(entry)
        if mid and not entry.get("hidden"):
            click.echo(mid)
