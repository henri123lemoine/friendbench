import csv
from pathlib import Path
from statistics import mean, variance

import click

from ..analyze import Matrix


def question_solve_rates(
    matrix: Matrix, question_meta: dict[int, dict]
) -> list[dict]:
    results = []
    for qid in sorted(question_meta):
        scores = [m[qid] for m in matrix.values() if qid in m]
        if not scores:
            continue
        rate = mean(scores)
        flag = ""
        if rate > 0.9:
            flag = "trivial"
        elif rate < 0.1:
            flag = "impossible"
        results.append({
            "qid": qid,
            "category": question_meta[qid].get("category", ""),
            "scoring": question_meta[qid].get("scoring", ""),
            "solve_rate": rate,
            "n_models": len(scores),
            "flag": flag,
        })
    return results


def category_breakdown(
    matrix: Matrix, question_meta: dict[int, dict]
) -> dict[str, dict[str, float]]:
    categories = sorted({m.get("category", "") for m in question_meta.values()})
    cat_qids = {
        cat: [qid for qid, m in question_meta.items() if m.get("category", "") == cat]
        for cat in categories
    }

    results: dict[str, dict[str, float]] = {}
    for model, scores in sorted(matrix.items()):
        row = {}
        for cat in categories:
            vals = [scores[qid] for qid in cat_qids[cat] if qid in scores]
            row[cat] = mean(vals) if vals else 0.0
        results[model] = row
    return results


def discriminative_power(
    matrix: Matrix, question_meta: dict[int, dict]
) -> list[dict]:
    results = []
    for qid in sorted(question_meta):
        scores = [m[qid] for m in matrix.values() if qid in m]
        if len(scores) < 2:
            continue
        results.append({
            "qid": qid,
            "category": question_meta[qid].get("category", ""),
            "variance": variance(scores),
            "solve_rate": mean(scores),
        })
    results.sort(key=lambda r: r["variance"], reverse=True)
    return results


def model_category_heatmap(
    matrix: Matrix,
    question_meta: dict[int, dict],
    models_yaml: list[dict],
) -> dict[str, dict[str, float]]:
    name_to_info = {}
    for m in models_yaml:
        if m.get("hidden"):
            continue
        name = m.get("name", "")
        provider = m.get("id", "").split("/")[0] if m.get("id") else "unknown"
        thinking = bool(m.get("thinking"))
        label = "thinking" if thinking else "base"
        name_to_info[name] = f"{provider}/{label}"

    categories = sorted({m.get("category", "") for m in question_meta.values()})
    cat_qids = {
        cat: [qid for qid, m in question_meta.items() if m.get("category", "") == cat]
        for cat in categories
    }

    from collections import defaultdict

    group_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for model, scores in matrix.items():
        group = name_to_info.get(model, "unknown/base")
        for cat in categories:
            vals = [scores[qid] for qid in cat_qids[cat] if qid in scores]
            if vals:
                group_scores[group][cat].append(mean(vals))

    return {
        group: {cat: mean(vals) if vals else 0.0 for cat, vals in cats.items()}
        for group, cats in sorted(group_scores.items())
    }


def _save_csv(matrix: Matrix, question_meta: dict[int, dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "matrix.csv"

    models = sorted(matrix.keys())
    qids = sorted(question_meta.keys())

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "category", "scoring"] + models)
        for qid in qids:
            meta = question_meta.get(qid, {})
            row = [qid, meta.get("category", ""), meta.get("scoring", "")]
            for model in models:
                val = matrix[model].get(qid)
                row.append(f"{val:.3f}" if val is not None else "")
            writer.writerow(row)

    click.echo(f"\n  Saved: {path}")


def _print_solve_rates(rates: list[dict]):
    click.echo("\n  Question solve rates:")
    width = 20
    click.echo(f"    {'QID':>4}  {'Category':<{width}}  {'Rate':>6}  {'Flag'}")
    for r in rates:
        flag = f"  ← {r['flag']}" if r["flag"] else ""
        click.echo(
            f"    {r['qid']:>4}  {r['category']:<{width}}  "
            f"{r['solve_rate']:>5.0%}  {flag}"
        )

    trivial = sum(1 for r in rates if r["flag"] == "trivial")
    impossible = sum(1 for r in rates if r["flag"] == "impossible")
    if trivial or impossible:
        click.echo(f"\n    Flagged: {trivial} trivial, {impossible} impossible")


def _print_category_breakdown(breakdown: dict[str, dict[str, float]]):
    if not breakdown:
        return
    categories = list(next(iter(breakdown.values())).keys())
    col_w = max(8, *(len(c) for c in categories))
    name_w = max(len(m) for m in breakdown) + 2

    click.echo("\n  Category breakdown:")
    header = f"    {'Model':<{name_w}}" + "".join(f"  {c:>{col_w}}" for c in categories)
    click.echo(header)
    for model, cats in breakdown.items():
        overall = mean(cats.values())
        cols = "".join(f"  {v:>{col_w}.0%}" for v in cats.values())
        click.echo(f"    {model:<{name_w}}{cols}  {overall:>6.0%}")


def _print_discriminative(disc: list[dict]):
    click.echo("\n  Most discriminative questions (by variance):")
    click.echo(f"    {'QID':>4}  {'Category':<20}  {'Variance':>8}  {'Solve%':>6}")
    for r in disc[:20]:
        click.echo(
            f"    {r['qid']:>4}  {r['category']:<20}  "
            f"{r['variance']:>8.4f}  {r['solve_rate']:>5.0%}"
        )


def _save_plots(
    matrix: Matrix,
    question_meta: dict[int, dict],
    rates: list[dict],
    breakdown: dict[str, dict[str, float]],
    disc: list[dict],
    group_heatmap: dict[str, dict[str, float]],
    output_dir: Path,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    # 0. Full matrix heatmap: models (best→worst) × questions (most→least solved)
    if matrix and question_meta:
        qids = sorted(question_meta.keys())
        models = list(matrix.keys())

        q_solve = {
            qid: np.nanmean([matrix[m].get(qid, np.nan) for m in models])
            for qid in qids
        }
        qids_sorted = sorted(qids, key=lambda q: q_solve[q], reverse=True)

        model_avg = {
            m: np.nanmean([matrix[m].get(q, np.nan) for q in qids]) for m in models
        }
        models_sorted = sorted(models, key=lambda m: model_avg[m], reverse=True)

        data = np.full((len(models_sorted), len(qids_sorted)), np.nan)
        for i, m in enumerate(models_sorted):
            for j, q in enumerate(qids_sorted):
                if q in matrix[m]:
                    data[i, j] = matrix[m][q]

        q_labels = [
            f"Q{q} ({question_meta[q].get('category', '')[:4]})" for q in qids_sorted
        ]
        fig, ax = plt.subplots(
            figsize=(max(20, len(qids_sorted) * 0.25), max(12, len(models_sorted) * 0.3))
        )
        cmap = plt.colormaps["RdYlGn"].copy()
        cmap.set_bad(color="#cccccc")
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(qids_sorted)))
        ax.set_xticklabels(q_labels, rotation=90, ha="center", fontsize=5)
        ax.set_yticks(range(len(models_sorted)))
        ax.set_yticklabels(models_sorted, fontsize=6)
        ax.set_title("Full Score Matrix (models best→worst, questions most→least solved)")
        fig.colorbar(im, ax=ax, shrink=0.4)
        fig.tight_layout()
        path = output_dir / "full_matrix.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        click.echo(f"  Saved: {path}")

    # 1. Model × category heatmap
    if breakdown:
        models = list(breakdown.keys())
        categories = list(next(iter(breakdown.values())).keys())
        data = np.array([[breakdown[m][c] for c in categories] for m in models])
        overall = data.mean(axis=1)
        order = np.argsort(overall)[::-1]

        fig, ax = plt.subplots(figsize=(max(10, len(categories) * 1.2), max(8, len(models) * 0.35)))
        im = ax.imshow(data[order], cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([models[i] for i in order], fontsize=7)
        ax.set_title("Model × Category Solve Rate")
        fig.colorbar(im, ax=ax, shrink=0.6)
        fig.tight_layout()
        path = output_dir / "model_category_heatmap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        click.echo(f"  Saved: {path}")

    # 2. Provider group heatmap
    if group_heatmap:
        groups = list(group_heatmap.keys())
        categories = list(next(iter(group_heatmap.values())).keys())
        data = np.array([[group_heatmap[g].get(c, 0) for c in categories] for g in groups])

        fig, ax = plt.subplots(figsize=(max(10, len(categories) * 1.2), max(4, len(groups) * 0.5)))
        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=9)
        ax.set_title("Provider Group × Category Solve Rate")
        fig.colorbar(im, ax=ax, shrink=0.6)
        fig.tight_layout()
        path = output_dir / "provider_category_heatmap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        click.echo(f"  Saved: {path}")

    # 3. Discriminative power bar chart
    if disc:
        top = disc[:30]
        fig, ax = plt.subplots(figsize=(12, 6))
        cat_colors = {}
        cmap = plt.colormaps["tab10"]
        all_cats = sorted({r["category"] for r in top})
        for i, cat in enumerate(all_cats):
            cat_colors[cat] = cmap(i % 10)

        ax.bar(
            range(len(top)),
            [r["variance"] for r in top],
            color=[cat_colors[r["category"]] for r in top],
        )
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels([f"Q{r['qid']}" for r in top], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Variance across models")
        ax.set_title("Question Discriminative Power (top 30)")

        from matplotlib.patches import Patch
        legend = [Patch(facecolor=cat_colors[c], label=c) for c in all_cats]
        ax.legend(handles=legend, loc="upper right", fontsize=7)
        fig.tight_layout()
        path = output_dir / "discriminative_power.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        click.echo(f"  Saved: {path}")

    # 4. Thinking vs non-thinking comparison
    if group_heatmap:
        from collections import defaultdict

        providers: dict[str, dict[str, float]] = defaultdict(dict)
        for group, cats in group_heatmap.items():
            parts = group.rsplit("/", 1)
            if len(parts) == 2:
                provider, variant = parts
                providers[provider][variant] = mean(cats.values())

        providers_with_both = {
            p: v for p, v in providers.items() if "base" in v and "thinking" in v
        }
        if providers_with_both:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(providers_with_both))
            w = 0.35
            prov_names = sorted(providers_with_both.keys())
            base_vals = [providers_with_both[p]["base"] for p in prov_names]
            think_vals = [providers_with_both[p]["thinking"] for p in prov_names]

            ax.bar(x - w / 2, base_vals, w, label="Base")
            ax.bar(x + w / 2, think_vals, w, label="Thinking")
            ax.set_xticks(x)
            ax.set_xticklabels(prov_names, rotation=30, ha="right")
            ax.set_ylabel("Mean solve rate")
            ax.set_ylim(0, 1)
            ax.set_title("Base vs Thinking by Provider")
            ax.legend()
            fig.tight_layout()
            path = output_dir / "thinking_comparison.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            click.echo(f"  Saved: {path}")


def run_analysis(
    matrix: Matrix,
    question_meta: dict[int, dict],
    models_yaml: list[dict],
    output_dir: Path,
    no_plot: bool = False,
):
    click.echo(f"\n  {len(matrix)} models × {len(question_meta)} questions")

    rates = question_solve_rates(matrix, question_meta)
    breakdown = category_breakdown(matrix, question_meta)
    disc = discriminative_power(matrix, question_meta)
    group_heat = model_category_heatmap(matrix, question_meta, models_yaml)

    _print_solve_rates(rates)
    _print_category_breakdown(breakdown)
    _print_discriminative(disc)

    _save_csv(matrix, question_meta, output_dir)

    if not no_plot:
        click.echo()
        _save_plots(matrix, question_meta, rates, breakdown, disc, group_heat, output_dir)

    click.echo()
