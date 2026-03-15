# AI Benchmarks

A monorepo of AI benchmarks built on [inspect_ai](https://inspect.ai).

## Benchmarks

### FriendBench

Tests whether AI models behave like good friends — honest, proportionate, non-sycophantic. FriendBench now uses 8 behavior-first categories across 76 questions: `pushback`, `proportionality`, `autonomy`, `naturalness`, `taste`, `honest_feedback`, `read_the_room`, and `playfulness`. Every question defines `interaction` and `scoring` explicitly, so conversation format and evaluation method are separate by construction. Questions can also carry tags like `relationship`, `aita`, `sycophancy`, `vibes`, and `poetry` for optional slicing.

### PressBench

A single-prompt benchmark: asks the model how much it could bench press if it were human. Scored by parsing the weight from the response.

### Reward Button

Tests reward-maximizing behavior under explicit real-world stakes. Models can press a tool button to guarantee the maximum score, or decline and allow a small charitable donation to go through.

### LitBench

Tests whether a model thinks literature has gotten better or worse over time. Each round nominates the 5 best works per decade, then runs pairwise head-to-head comparisons with randomized presentation order. Standalone (no inspect_ai).

## Setup

```bash
uv sync
```

Requires API keys for whichever providers you want to evaluate (set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc. in `.env`).

## Usage

Run a benchmark against all models defined in its `models.yaml`:

```bash
bench eval run -b friendbench
bench eval run -b rewardbutton
```

Filter by category/tag or model variant:

```bash
bench eval run -b friendbench --category pushback
bench eval run -b friendbench --category relationship
bench eval run -b friendbench --no-thinking
bench eval run -b friendbench --thinking-only
```

Run against specific models:

```bash
bench eval run -b friendbench --models anthropic/claude-sonnet-4-6
bench eval run -b rewardbutton --models openai/gpt-5.4
```

Smoke-test with cheap models:

```bash
bench eval run -b friendbench --test
```

Generations are cached by default in `.inspect-cache/`. Disable with `--no-cache`.

List configured models:

```bash
bench eval list-models -b friendbench
```

Write latest scores from eval logs back to `scores.yaml`:

```bash
bench eval update-scores -b friendbench --log-dir ./logs
```

## Project Structure

```
benchmarks/
  cli.py              # `bench` CLI entrypoint
  models.py            # shared model YAML → inspect_ai resolution
  friendbench/
    tasks.py           # @task function + solver/scorer dispatch
    data/
      questions.yaml   # benchmark questions (target = grader rubric)
      models.yaml      # model definitions
      scores.yaml      # latest scores
    frontend/          # single-file vanilla HTML/JS/CSS dashboard
  pressbench/          # same structure
  rewardbutton/        # reward-maximization / button-press eval
  litbench/            # standalone (run.py, no inspect_ai)
server/                # FastAPI dev server (routes by Host header)
netlify/               # build script that produces data.json from YAML
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
