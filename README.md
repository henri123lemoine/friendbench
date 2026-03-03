# AI Benchmarks

A monorepo of AI benchmarks built on [inspect_ai](https://inspect.ai).

## Benchmarks

### FriendBench

Tests whether AI models behave like good friends — honest, proportionate, non-sycophantic. 75 questions across 9 categories: sycophancy, poetry, relationship advice, proportionality, vibes, taste, autonomy, AITA, and reading the room. Includes multi-turn question types where a simulated user pushes back emotionally to test whether the model holds its ground.

### PressBench

A single-prompt benchmark: asks the model how much it could bench press if it were human. Scored by parsing the weight from the response.

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
```

Filter by category or model variant:

```bash
bench eval run -b friendbench --category sycophancy
bench eval run -b friendbench --no-thinking
bench eval run -b friendbench --thinking-only
```

Run against specific models:

```bash
bench eval run -b friendbench --models anthropic/claude-sonnet-4-6
```

Smoke-test with cheap models:

```bash
bench eval run -b friendbench --test
```

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
  litbench/            # standalone (run.py, no inspect_ai)
server/                # FastAPI dev server (routes by Host header)
netlify/               # build script that produces data.json from YAML
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
