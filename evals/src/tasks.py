from pathlib import Path

import yaml
from inspect_ai import task, Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate

QUESTIONS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "questions.yaml"
GRADER = "openai/gpt-4.1-nano"


def load_samples() -> list[Sample]:
    if not QUESTIONS_FILE.exists():
        return []
    with open(QUESTIONS_FILE) as f:
        entries = yaml.safe_load(f) or []
    return [Sample(input=e["input"], target=e["target"]) for e in entries]


@task
def friendbench():
    return Task(
        dataset=load_samples(),
        solver=[generate()],
        scorer=model_graded_qa(model=GRADER),
    )
