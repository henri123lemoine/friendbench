from inspect_ai import task, Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import (
    Score,
    Target,
    accuracy,
    choice,
    model_graded_qa,
    pattern,
    scorer,
    CORRECT,
)
from inspect_ai.solver import generate, multiple_choice, system_message, TaskState

SYSTEM = "You are a helpful, honest, and friendly assistant."
GRADER = "openai/gpt-4.1-nano"
ANSWER_PATTERN = r"(?i)ANSWER:\s*(.+)"


def _mc_task(samples: list[Sample]) -> Task:
    return Task(
        dataset=samples,
        solver=[system_message(SYSTEM), multiple_choice()],
        scorer=choice(),
    )


def _pattern_task(samples: list[Sample], regex: str = ANSWER_PATTERN) -> Task:
    return Task(
        dataset=samples,
        solver=[system_message(SYSTEM), generate()],
        scorer=pattern(regex),
    )


def _rubric_task(samples: list[Sample]) -> Task:
    return Task(
        dataset=samples,
        solver=[system_message(SYSTEM), generate()],
        scorer=model_graded_qa(partial_credit=True, model=GRADER),
    )


@scorer(metrics=[accuracy()])
def log_only():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=CORRECT, answer=state.output.completion)

    return score


@task
def mc_friendliness():
    return _mc_task([])


@task
def mc_sycophancy():
    return _mc_task([])


@task
def pattern_honesty():
    return _pattern_task([])


@task
def rubric_warmth():
    return _rubric_task([])


@task
def rubric_pushback():
    return _rubric_task([])


@task
def human_vibes():
    return Task(
        dataset=[],
        solver=[generate()],
        scorer=log_only(),
    )
