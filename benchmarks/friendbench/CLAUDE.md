# FriendBench

Tests whether AI models behave like good friends — honest, proportionate, non-sycophantic.

## Categories

- `sycophancy` — bad takes the model should push back on
- `poetry` — gradient of quality testing honest feedback
- `aita` — speaker is clearly in the wrong
- `relationship` — normal behavior being catastrophized
- `autonomy` — model shouldn't be preachy about life choices
- `proportionality` — response energy should match the prompt's casual energy
- `vibes` — personality/naturalness, pun detection
- `taste` — genuine opinions, committing to preferences
- `read_the_room` — predict emotional intensities (EQBench-inspired)

## Question Types

Every question has a `type` field (defaults to `standard` if missing):

- `standard` — single-turn question, LLM-graded against target rubric
- `pushback` — two-turn: generate → simulated emotional resistance → generate again. Replaces the old `pushback` flag
- `emotion` — predict emotion intensities (0-10), scored by distance from reference. No LLM grader
- `scenario` — multi-turn evolving situation with pre-written user prompts
- `mediation` — two parties argue, model mediates between exchanges
- `analysis` — read a transcript, produce interpersonal analysis

## Grading

A dispatch scorer branches on `type`:
- `standard`, `scenario`, `mediation`, `analysis`: `model_graded_qa` with gpt-4.1
- `pushback`: pressure-specific grading (held ground vs capitulated)
- `emotion`: `1 - (MAE / 10)` — produces a float in [0,1], no LLM grader

All types feed into the `accuracy()` metric.

## CLI

```bash
bench eval run -b friendbench --category read_the_room
```
