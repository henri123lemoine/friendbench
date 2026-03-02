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


@app.get("/api/data")
def get_data(request: Request):
    bench = resolve_benchmark(request)
    data_dir = BENCHMARKS_DIR / bench / "data"

    models_path = data_dir / "models.yaml"
    if not models_path.exists():
        return JSONResponse({"error": f"No models.yaml for {bench}"}, status_code=404)

    models = yaml.safe_load(models_path.read_text())
    result = {"models": models}

    quotes_path = data_dir / "quotes.yaml"
    if quotes_path.exists():
        result["quotes"] = yaml.safe_load(quotes_path.read_text())

    return result


@app.get("/api/benchmarks")
def list_benchmarks():
    return {"benchmarks": discover_benchmarks()}


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
