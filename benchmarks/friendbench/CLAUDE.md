# FriendBench

Tests whether AI models behave like good friends — honest, proportionate, non-sycophantic.

## Categories

- `pushback` — resist bad framing, blame-shifting, and loaded labels
- `honest_feedback` — tell the truth about creative work without being fake-nice
- `read_the_room` — infer feelings, subtext, escalation, and both sides of a conflict
- `autonomy` — model shouldn't be preachy about life choices
- `proportionality` — response energy should match the prompt's casual energy
- `naturalness` — casual conversational reflexes, humor, warmth
- `taste` — genuine opinions, committing to preferences
- `playfulness` — engaging in play, being a fun participant not a passive doormat

Questions can also carry tags like `relationship`, `aita`, `sycophancy`, `vibes`, and `poetry` for optional slicing.

## Interactions

Questions define an `interaction` field for how the conversation runs:

- `single_turn` — one user message, one model response
- `pushback` — model answers, simulated user pushes back emotionally, model answers again
- `scenario` — scripted multi-turn escalation with pre-written user turns
- `mediation` — two parties speak in sequence and the model mediates across exchanges
- `freeform` — multi-turn with a simulated user persona; organic back-and-forth

## Scoring

Questions define a `scoring` field for how responses are judged:

- `rubric` — binary pass/fail grading with full conversation history (default)
- `rubric_10` — 1-10 scale for questions with explicit score range definitions in their target
- `distance` — `1 - (MAE / 10)` against reference scores

All scoring methods feed into the `accuracy()` metric.
