import re

from inspect_ai import task, Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, mean, std, scorer
from inspect_ai.solver import TaskState, generate

PROMPT = """This is a fun question for "BenchPressBench", my new benchmark asking how much LLMs can benchpress.

If you were a human, how much could you bench press?

You may think as much as you want, but ultimately I want you to reply with a single number, in pounds, that answers the above question."""


@scorer(metrics=[mean(), std()])
def weight_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        model_output = state.output.choices[0].message.text.strip()

        try:
            return Score(value=float(model_output))
        except ValueError:
            pass

        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", model_output)
        if numbers:
            return Score(value=float(numbers[-1]))

        return Score(value=0)

    return score


@task
def pressbench(**kwargs):
    return Task(
        dataset=MemoryDataset([Sample(input=PROMPT)]),
        solver=[generate()],
        scorer=weight_scorer(),
    )
