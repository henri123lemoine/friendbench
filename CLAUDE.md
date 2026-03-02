# AI Benchmark Monorepo

FriendBench, PressBench, LitBench. Built on [inspect_ai](https://inspect.ai).

## Structure

Each benchmark lives in `benchmarks/<name>/` with `tasks.py`, `data/`, and `frontend/`. The `@task` function name must match the directory name. A benchmark is discovered by the server/CLI if it has `data/models.yaml`.

LitBench is standalone (no inspect_ai, just `run.py`).

## Non-obvious things

- `target` in questions.yaml is a **grader rubric**, not a correct answer
- Thinking variants are separate model entries: same `id`, different `name`, with a `generation_config`
- Frontends are single HTML files with no build step — vanilla JS, inline CSS, SVG charts
- Dev server routes by Host header (see `DOMAIN_MAP` in `server/main.py`)
- Netlify function is currently hardcoded to friendbench

## CLI

```bash
bench eval run -b friendbench
bench eval run -b friendbench --category sycophancy --multi-turn
bench eval list-models -b friendbench
```
