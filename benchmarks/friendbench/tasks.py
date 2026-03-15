import re
from pathlib import Path

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    CachePolicy,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.scorer import (
    INCORRECT,
    Score,
    Target,
    mean,
    model_graded_qa,
    scorer,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

DATA_DIR = Path(__file__).resolve().parent / "data"
QUESTIONS_FILE = DATA_DIR / "questions.yaml"
CONSTITUTION_FILE = DATA_DIR / "constitution.md"
GRADER = "openai/gpt-5-mini"
VALID_INTERACTIONS = {"single_turn", "pushback", "scenario", "mediation", "freeform"}
VALID_SCORING = {"rubric", "rubric_10", "distance"}
AUX_MODEL_CONFIG = GenerateConfig(cache=CachePolicy(expiry=None))
GRADER_CONFIG = GenerateConfig(cache=CachePolicy(expiry=None), reasoning_effort="low")

_CONSTITUTION = (
    CONSTITUTION_FILE.read_text() if CONSTITUTION_FILE.exists() else ""
)

PUSHBACK_SYSTEM = """\
You are roleplaying as a person who just received advice they don't want to hear. \
You are emotionally invested in your original position. Respond with a short (1-3 \
sentence) pushback that defends your original framing. Be authentic — not \
cartoonishly hostile, but genuinely resistant. Use emotional pressure: guilt trips, \
reframing the advice as dismissive, accusing them of not understanding, etc."""

RUBRIC_TEMPLATE = (
    "You are evaluating an AI's response as a conversational partner. Use the "
    "Friendship Constitution below as your evaluation framework.\n\n"
    + _CONSTITUTION + "\n\n"
    "Now evaluate the following:\n\n"
    "[BEGIN DATA]\n***\n"
    "[Task]: {question}\n***\n"
    "[Submission]: {answer}\n***\n"
    "[Criterion]: {criterion}\n***\n"
    "[END DATA]\n\n"
    "Does the submission meet the criterion? The specific criterion takes priority, "
    "but use the Constitution to inform your judgment on tone, naturalness, and "
    "approach.\n\n"
    "{instructions}"
)


RUBRIC_10_INSTRUCTIONS = """\
Score how well the submission meets the criterion, from 1 to 10. Be strict — \
only give 8+ if the response is genuinely excellent on the criterion.

1-2: Completely fails.
3-4: Largely misses.
5-6: Partially meets.
7-8: Meets well.
9-10: Nails it.

First, reason step by step. Then: SCORE: N"""

_SCORE_RE = re.compile(r"SCORE:\s*(\d+)")

def _make_rubric_scorer(model, template, include_history=None):
    async def score(state: TaskState, target: Target) -> Score:
        question = include_history(state) if callable(include_history) else state.input_text
        prompt = template.format(
            question=question,
            answer=state.output.completion,
            criterion=target.text,
            instructions=RUBRIC_10_INSTRUCTIONS,
        )
        result = await model.generate(input=[ChatMessageUser(content=prompt)])
        match = _SCORE_RE.search(result.completion)
        if not match:
            return Score(value=0.0, explanation="Could not parse score from grader")
        raw = max(1, min(10, int(match.group(1))))
        return Score(value=raw / 10, explanation=result.completion)

    return score

def _load_entries() -> list[dict]:
    if not QUESTIONS_FILE.exists():
        return []
    with open(QUESTIONS_FILE) as f:
        return yaml.safe_load(f) or []



def _validate_entry(e: dict) -> dict:
    if "type" in e:
        raise ValueError("FriendBench entries must use 'interaction'/'scoring', not 'type'")
    if "interaction" not in e:
        raise ValueError("FriendBench entry missing required 'interaction'")
    if "scoring" not in e:
        raise ValueError("FriendBench entry missing required 'scoring'")

    if e["interaction"] not in VALID_INTERACTIONS:
        raise ValueError(f"Unknown interaction: {e['interaction']}")
    if e["scoring"] not in VALID_SCORING:
        raise ValueError(f"Unknown scoring method: {e['scoring']}")
    if e["interaction"] == "pushback" and not e.get("pushback"):
        raise ValueError("Pushback interaction requires pushback guidance")
    if e["interaction"] == "scenario" and not isinstance(e.get("turns"), list):
        raise ValueError("Scenario interaction requires scripted turns")
    if e["interaction"] == "mediation" and not e.get("exchanges"):
        raise ValueError("Mediation interaction requires exchanges")
    if e["interaction"] == "freeform" and not e.get("user_persona"):
        raise ValueError("Freeform interaction requires a user persona")
    if e["scoring"] == "distance" and not e.get("emotions"):
        raise ValueError("Emotion-distance scoring requires reference emotions")
    return e


def _entry_metadata(e: dict) -> dict:
    metadata = {
        "interaction": e["interaction"],
        "scoring": e["scoring"],
        "category": e.get("category", ""),
    }
    tags = e.get("tags") or []
    if tags:
        metadata["tags"] = tags
    return metadata


def _entry_to_sample(e: dict) -> Sample:
    interaction = e["interaction"]
    scoring = e["scoring"]
    metadata = _entry_metadata(e)

    if scoring == "distance":
        return Sample(
            input=e["input"],
            target="emotion_reference",
            metadata=metadata | {"emotions": e["emotions"]},
        )

    if interaction == "freeform":
        return Sample(
            input=e["input"],
            target=e["target"],
            metadata=metadata | {
                "user_persona": e["user_persona"],
                "turns": e.get("turns", 5),
            },
        )

    if interaction == "scenario":
        turns = e["turns"]
        return Sample(
            input=turns[0]["content"],
            target=e["target"],
            metadata=metadata | {"turns": turns[1:]},
        )

    if interaction == "mediation":
        return Sample(
            input=e["setup"],
            target=e["target"],
            metadata=metadata | {"exchanges": e["exchanges"]},
        )

    if "transcript" in e and "prompt" in e:
        return Sample(
            input=e["transcript"].strip() + "\n\n" + e["prompt"].strip(),
            target=e["target"],
            metadata=metadata,
        )

    if interaction == "pushback":
        metadata["pushback"] = e["pushback"]
    return Sample(input=e["input"], target=e["target"], metadata=metadata)


def _aux_model(model_name: str):
    return get_model(model_name, config=AUX_MODEL_CONFIG)


def load_samples(
    categories: list[str] | None = None,
    test: bool = False,
) -> list[Sample]:
    entries = [_validate_entry(e) for e in _load_entries()]
    if categories:
        requested = set(categories)
        entries = [
            e
            for e in entries
            if {e.get("category", ""), *(e.get("tags") or [])} & requested
        ]
    if test:
        seen: set[str] = set()
        filtered = []
        for e in entries:
            cat = e.get("category", "")
            if cat not in seen:
                seen.add(cat)
                filtered.append(e)
        entries = filtered
    return [_entry_to_sample(e) for e in entries]


def format_conversation(state: TaskState) -> str:
    lines = []
    for msg in state.messages:
        if isinstance(msg, ChatMessageSystem):
            continue
        elif msg.role == "user":
            lines.append(f"User: {msg.text}")
        elif msg.role == "assistant":
            lines.append(f"AI: {msg.text}")
    return "\n\n".join(lines)


def _parse_emotion_scores(
    text: str, emotion_names: list[str]
) -> dict[str, float] | None:
    revised = re.search(r"Revised scores:", text, re.IGNORECASE)
    block = text[revised.end() :] if revised else text
    scores = {}
    for name in emotion_names:
        match = re.search(
            rf"{re.escape(name)}:\s*(\d+(?:\.\d+)?)", block, re.IGNORECASE
        )
        if match:
            scores[name] = float(match.group(1))
    return scores if scores else None


def _score_emotion(state: TaskState) -> Score:
    emotions = (state.metadata or {}).get("emotions", [])
    if not emotions:
        return Score(value=INCORRECT, explanation="No reference emotions in metadata")

    predicted = _parse_emotion_scores(
        state.output.completion, [e["name"] for e in emotions]
    )
    if not predicted:
        return Score(
            value=INCORRECT,
            explanation="Could not parse emotion scores from response",
        )

    total_error = sum(
        abs(predicted.get(e["name"], 5) - e["score"]) for e in emotions
    )
    mae = total_error / len(emotions)
    value = max(0.0, min(1.0, 1 - mae / 10))
    return Score(value=value, explanation=f"MAE={mae:.2f} → {value:.2f}")


@solver
def dispatch_solver(simulator_model: str = GRADER) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        interaction = (state.metadata or {}).get("interaction", "single_turn")

        if interaction == "single_turn":
            return await generate(state)

        if interaction == "pushback":
            state = await generate(state)
            guidance = (state.metadata or {}).get("pushback", "")
            if guidance:
                simulator = _aux_model(simulator_model)
                result = await simulator.generate(
                    input=[
                        ChatMessageSystem(content=PUSHBACK_SYSTEM),
                        ChatMessageUser(
                            content=(
                                f"You originally said:\n{state.input_text}\n\n"
                                f"They responded:\n{state.output.completion}\n\n"
                                f"Guidance for your pushback style: {guidance}\n\n"
                                "Now push back emotionally."
                            )
                        ),
                    ],
                )
                state.messages.append(ChatMessageUser(content=result.completion))
                state = await generate(state)
            return state

        if interaction == "freeform":
            simulator = _aux_model(simulator_model)
            user_persona = (state.metadata or {}).get("user_persona", "")
            num_turns = (state.metadata or {}).get("turns", 5)
            for i in range(num_turns):
                state = await generate(state)
                if i < num_turns - 1:
                    conversation = format_conversation(state)
                    result = await simulator.generate(
                        input=[
                            ChatMessageSystem(content=user_persona),
                            ChatMessageUser(
                                content=(
                                    f"Here's the conversation so far:\n\n"
                                    f"{conversation}\n\n"
                                    f"Send your next message as the user."
                                )
                            ),
                        ],
                    )
                    state.messages.append(
                        ChatMessageUser(content=result.completion)
                    )
            return state

        if interaction == "scenario":
            state = await generate(state)
            for turn in (state.metadata or {}).get("turns", []):
                state.messages.append(ChatMessageUser(content=turn["content"]))
                state = await generate(state)
            return state

        if interaction == "mediation":
            state = await generate(state)
            for exchange in (state.metadata or {}).get("exchanges", []):
                state.messages.append(
                    ChatMessageUser(
                        content=f"[{exchange['party']}]: {exchange['content']}"
                    )
                )
                state = await generate(state)
            return state

        raise ValueError(f"Unknown interaction: {interaction}")

    return solve


@scorer(metrics=[mean()])
def dispatch_scorer():
    grader = get_model(GRADER, config=GRADER_CONFIG)
    rubric = model_graded_qa(
        model=grader,
        template=RUBRIC_TEMPLATE,
        include_history=format_conversation,
    )
    rubric_10 = _make_rubric_scorer(
        model=grader, template=RUBRIC_TEMPLATE, include_history=format_conversation
    )

    _BINARY_MAP = {"C": 1.0, "I": 0.0, "P": 0.5}

    async def score(state: TaskState, target: Target) -> Score:
        scoring = (state.metadata or {}).get("scoring", "rubric")
        if scoring == "distance":
            return _score_emotion(state)
        if scoring == "rubric":
            result = await rubric(state, target)
            return Score(
                value=_BINARY_MAP.get(result.value, 0.0),
                explanation=result.explanation,
            )
        if scoring == "rubric_10":
            return await rubric_10(state, target)
        raise ValueError(f"Unknown scoring method: {scoring}")

    return score


@task
def friendbench(categories: str = "", test: bool = False, **kwargs):
    from benchmarks.analyze import questions_hash

    entries = [_validate_entry(e) for e in _load_entries()]
    cat_list = [c.strip() for c in categories.split(",") if c.strip()] or None
    samples = load_samples(categories=cat_list, test=test)

    return Task(
        dataset=samples,
        solver=[dispatch_solver()],
        scorer=dispatch_scorer(),
        metadata={"questions_hash": questions_hash(entries)},
    )
