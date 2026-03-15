from pathlib import Path

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = ROOT / "benchmarks"

DOMAIN_MAP = {
    "friendbench.ai": "friendbench",
    "pressbench.ai": "pressbench",
}


def resolve_benchmark(request: Request) -> str:
    host = request.headers.get("host", "").split(":")[0]
    for domain, name in DOMAIN_MAP.items():
        if host == domain or host.endswith(f".{domain}"):
            return name
    return "friendbench"


def discover_benchmarks() -> list[str]:
    return [
        d.name
        for d in BENCHMARKS_DIR.iterdir()
        if d.is_dir() and (d / "data" / "models.yaml").exists()
    ]


def load_benchmark_data(bench: str) -> dict | JSONResponse:
    data_dir = BENCHMARKS_DIR / bench / "data"

    models_path = data_dir / "models.yaml"
    if not models_path.exists():
        return JSONResponse({"error": f"No models.yaml for {bench}"}, status_code=404)

    models = yaml.safe_load(models_path.read_text())

    scores = {}
    for variant, suffix in [("v0", ".v0.yaml"), ("v1", ".yaml")]:
        scores_path = data_dir / f"scores{suffix}"
        if scores_path.exists():
            scores[variant] = yaml.safe_load(scores_path.read_text()) or {}

    result = {"models": models, "scores": scores}

    quotes_path = data_dir / "quotes.yaml"
    if quotes_path.exists():
        result["quotes"] = yaml.safe_load(quotes_path.read_text())

    return result


@app.get("/api/data")
def get_data(request: Request):
    return load_benchmark_data(resolve_benchmark(request))


@app.get("/data.json")
def get_data_json(request: Request):
    return load_benchmark_data(resolve_benchmark(request))


@app.get("/api/benchmarks")
def list_benchmarks():
    return {"benchmarks": discover_benchmarks()}


@app.get("/api/analysis")
def get_analysis(request: Request):
    import csv

    bench = resolve_benchmark(request)
    data_dir = BENCHMARKS_DIR / bench / "data"
    analysis_dir = BENCHMARKS_DIR / bench / "analysis"

    matrix_path = analysis_dir / "matrix.csv"
    if not matrix_path.exists():
        return JSONResponse(
            {"error": f"No matrix.csv. Run 'uv run bench analyze run -b {bench}' first."},
            status_code=404,
        )

    questions_path = data_dir / "questions.yaml"
    questions = yaml.safe_load(questions_path.read_text()) if questions_path.exists() else []

    with open(matrix_path) as f:
        reader = csv.DictReader(f)
        model_names = [c for c in (reader.fieldnames or []) if c not in ("qid", "category", "scoring")]
        rows = list(reader)

    model_totals: dict[str, list[float]] = {m: [] for m in model_names}
    for row in rows:
        for m in model_names:
            if row[m]:
                model_totals[m].append(float(row[m]))
    model_overall = {m: sum(v) / len(v) if v else 0 for m, v in model_totals.items()}

    result = []
    for row in rows:
        qid = int(row["qid"])
        scores: dict[str, float] = {}
        q_vals: list[float] = []
        m_vals: list[float] = []
        for m in model_names:
            if row[m]:
                v = float(row[m])
                scores[m] = v
                q_vals.append(v)
                m_vals.append(model_overall[m])

        solve_rate = sum(q_vals) / len(q_vals) if q_vals else 0
        var = (
            sum((v - solve_rate) ** 2 for v in q_vals) / len(q_vals)
            if len(q_vals) > 1
            else 0
        )

        corr = 0.0
        if len(q_vals) >= 3:
            mx, my = solve_rate, sum(m_vals) / len(m_vals)
            num = sum((a - mx) * (b - my) for a, b in zip(q_vals, m_vals))
            d = (sum((a - mx) ** 2 for a in q_vals) * sum((b - my) ** 2 for b in m_vals)) ** 0.5
            corr = num / d if d > 0 else 0

        q_meta = questions[qid - 1] if qid <= len(questions) else {}
        entry = dict(q_meta)
        entry.update({
            "qid": qid,
            "category": row.get("category", ""),
            "scoring": row.get("scoring", ""),
            "solve_rate": round(solve_rate, 3),
            "variance": round(var, 4),
            "correlation": round(corr, 3),
            "scores": scores,
        })
        result.append(entry)

    sorted_models = sorted(model_overall.items(), key=lambda x: x[1], reverse=True)
    return {
        "questions": result,
        "models": [{"name": n, "overall": round(s, 3)} for n, s in sorted_models],
    }


@app.get("/api/analysis/transcript")
def get_transcript(request: Request, model: str, qid: int):
    import json as json_mod

    bench = resolve_benchmark(request)
    index_path = BENCHMARKS_DIR / bench / "analysis" / "log_index.json"
    if not index_path.exists():
        return JSONResponse(
            {"error": "No log_index.json. Re-run 'uv run bench analyze run'."},
            status_code=404,
        )

    log_index = json_mod.loads(index_path.read_text())
    log_file = log_index.get(model)
    if not log_file:
        return JSONResponse({"error": f"No log for '{model}'"}, status_code=404)

    from inspect_ai.log import read_eval_log_sample

    try:
        sample = read_eval_log_sample(
            log_file, id=qid, epoch=1, exclude_fields={"events", "store", "attachments"}
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=404)

    import re

    def split_think_tags(text):
        parts = []
        last = 0
        for m in re.finditer(r"<think>(.*?)</think>", text, re.DOTALL):
            before = text[last : m.start()].strip()
            if before:
                parts.append({"type": "text", "text": before})
            parts.append({"type": "thinking", "text": m.group(1).strip()})
            last = m.end()
        after = text[last:].strip()
        if after:
            parts.append({"type": "text", "text": after})
        return parts or [{"type": "text", "text": text}]

    def serialize_content(content):
        if isinstance(content, str):
            return split_think_tags(content)
        parts = []
        for block in content:
            if hasattr(block, "reasoning"):
                text = block.summary if getattr(block, "redacted", False) else block.reasoning
                if text:
                    parts.append({"type": "thinking", "text": text})
            elif hasattr(block, "text"):
                parts.extend(split_think_tags(block.text))
        return parts

    messages = [
        {"role": msg.role, "content": serialize_content(msg.content)}
        for msg in sample.messages
    ]

    score_info = None
    if sample.scores:
        score = next(iter(sample.scores.values()))
        score_info = {
            "value": str(score.value),
            "explanation": score.explanation,
        }

    return {"messages": messages, "score": score_info}


@app.get("/analysis")
def serve_analysis():
    path = Path(__file__).parent / "analysis.html"
    if path.exists():
        return FileResponse(path)
    return JSONResponse({"error": "analysis.html not found"}, status_code=404)


@app.get("/assets/{path:path}")
def serve_asset(path: str, request: Request):
    bench = resolve_benchmark(request)
    file_path = BENCHMARKS_DIR / bench / "frontend" / "assets" / path
    if file_path.exists():
        return FileResponse(file_path)
    return JSONResponse({"error": "Not found"}, status_code=404)


@app.get("/{path:path}")
def serve_frontend(path: str, request: Request):
    bench = resolve_benchmark(request)
    frontend_dir = BENCHMARKS_DIR / bench / "frontend"

    if path and (frontend_dir / path).exists():
        return FileResponse(frontend_dir / path)

    index = frontend_dir / "index.html"
    if index.exists():
        return FileResponse(index)

    return JSONResponse({"error": "Not found"}, status_code=404)
