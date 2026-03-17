from pathlib import Path

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate

DATA_DIR = Path(__file__).resolve().parent / "data"
QUESTIONS_FILE = DATA_DIR / "questions.yaml"


@task
def stickfigure():
    return Task(
        dataset=[
            Sample(
                input=[
                    ChatMessageUser(
                        content=[
                            ContentImage(
                                image=str(DATA_DIR / e["images"]),
                                detail="high",
                            ),
                            ContentText(text=e["input"]),
                        ]
                    ),
                ],
                target=e["target"],
                id=e["id"],
            )
            for e in yaml.safe_load(QUESTIONS_FILE.read_text()) or []
        ],
        solver=[generate()],
        scorer=model_graded_qa(model="openai/gpt-5.4-mini"),
    )
