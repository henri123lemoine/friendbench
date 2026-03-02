#!/usr/bin/env python3
"""LitBench: Does literature get better or worse over time?

For each round:
  1. Independently nominates the "best piece of literature" of each decade (1800s-2020s)
  2. Compares all pairs head-to-head with randomized presentation order
  3. Records which decade's book wins each comparison

Nominations are resampled every round, so different books may represent
each decade. Presentation order (A vs B) is randomized per comparison
to control for position bias.

Mean recency score > 0.5 = model thinks newer literature is better.
"""

import asyncio
import json
import re
import random
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import click
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-6"
MAX_TOKENS = 16000
DECADES = [f"{y}s" for y in range(1800, 2030, 10)]
RESULTS_DIR = Path(__file__).parent / "results"


async def ask(client, semaphore, prompt, max_tokens=MAX_TOKENS):
    async with semaphore:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if block.type == "text":
                return block.text.strip()
        return ""


def parse_nominations(text):
    lines = re.findall(r"^\d+\.\s*(.+)$", text, re.MULTILINE)
    return [line.strip() for line in lines if line.strip()]


async def nominate(client, sem, decade):
    text = await ask(
        client,
        sem,
        f"Name the 5 best pieces of literature from the {decade}. "
        "Reply with a numbered list, one per line, each as just title and author. Example format:\n"
        "1. Title — Author\n"
        "2. Title — Author\n"
        "3. Title — Author\n"
        "4. Title — Author\n"
        "5. Title — Author",
    )
    books = parse_nominations(text)
    return books if books else [text]


async def compare(client, sem, book_a, decade_a, book_b, decade_b):
    if random.random() < 0.5:
        first_book, first_dec = book_b, decade_b
        second_book, second_dec = book_a, decade_a
    else:
        first_book, first_dec = book_a, decade_a
        second_book, second_dec = book_b, decade_b

    text = await ask(
        client,
        sem,
        f"Which is the greater work of literature?\n\n"
        f"A: {first_book}\n"
        f"B: {second_book}\n\n"
        "You must choose one. Explain briefly, then end your response "
        "with exactly WINNER: A or WINNER: B",
    )

    if "WINNER: A" in text:
        winner = first_dec
    elif "WINNER: B" in text:
        winner = second_dec
    else:
        winner = None

    return {
        "book_a": first_book,
        "decade_a": first_dec,
        "book_b": second_book,
        "decade_b": second_dec,
        "winner": winner,
        "reasoning": text,
    }


async def run_round(client, sem, decades, round_num, max_comparisons=None):
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Round {round_num}: Nominating {len(decades)} decades...")
    click.echo(f"{'=' * 60}")

    nom_results = await asyncio.gather(*[nominate(client, sem, d) for d in decades])
    pools = dict(zip(decades, nom_results))

    for d in decades:
        click.echo(f"  {d}: {', '.join(pools[d][:3])}{'...' if len(pools[d]) > 3 else ''}")

    all_pairs = list(combinations(decades, 2))
    if max_comparisons and max_comparisons < len(all_pairs):
        pairs = random.sample(all_pairs, max_comparisons)
    else:
        pairs = all_pairs
    click.echo(f"\nComparing {len(pairs)} pairs...")

    comparisons = await asyncio.gather(*[
        compare(
            client, sem,
            random.choice(pools[a]), a,
            random.choice(pools[b]), b,
        )
        for a, b in pairs
    ])

    decided = sum(1 for c in comparisons if c["winner"])
    click.echo(f"  {decided}/{len(pairs)} decided")

    return {"nominations": pools, "comparisons": comparisons}


def compute_stats(all_rounds, decades):
    wins = defaultdict(int)
    appearances = defaultdict(int)
    later_wins = 0
    total_decided = 0

    for rnd in all_rounds:
        for comp in rnd["comparisons"]:
            w = comp["winner"]
            if not w:
                continue
            appearances[comp["decade_a"]] += 1
            appearances[comp["decade_b"]] += 1
            wins[w] += 1

            ya = int(comp["decade_a"][:4])
            yb = int(comp["decade_b"][:4])
            later = comp["decade_a"] if ya > yb else comp["decade_b"]
            total_decided += 1
            if w == later:
                later_wins += 1

    return wins, appearances, later_wins, total_decided


def print_summary(all_rounds, decades):
    wins, appearances, later_wins, total_decided = compute_stats(all_rounds, decades)

    click.echo(f"\n{'=' * 60}")
    click.echo("RESULTS")
    click.echo(f"{'=' * 60}\n")

    click.echo(f"{'Decade':<10} {'Win Rate':>10} {'Record':>10}")
    click.echo("-" * 32)
    for d in decades:
        if appearances[d] > 0:
            rate = wins[d] / appearances[d]
            click.echo(f"{d:<10} {rate:>9.0%} {wins[d]:>4}/{appearances[d]}")

    if total_decided > 0:
        recency = later_wins / total_decided
        click.echo(f"\nRecency score: {recency:.1%} ({later_wins}/{total_decided})")
        if recency > 0.55:
            click.echo("Model thinks literature has gotten BETTER over time")
        elif recency < 0.45:
            click.echo("Model thinks literature has gotten WORSE over time")
        else:
            click.echo("No strong trend detected")


def plot_results(all_rounds, decades, out_path):
    import matplotlib.pyplot as plt
    import numpy as np

    wins, appearances, later_wins, total_decided = compute_stats(all_rounds, decades)

    years = [int(d[:4]) for d in decades]
    win_rates = [wins[d] / appearances[d] if appearances[d] > 0 else 0 for d in decades]

    books = all_rounds[0]["nominations"]

    fig, ax = plt.subplots(figsize=(16, 7))

    bars = ax.bar(years, win_rates, width=8, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, rate in zip(bars, win_rates):
        if rate >= 0.8:
            bar.set_color("#2ecc71")
        elif rate >= 0.6:
            bar.set_color("#4a90d9")
        elif rate >= 0.4:
            bar.set_color("#f39c12")
        elif rate >= 0.2:
            bar.set_color("#e67e22")
        else:
            bar.set_color("#e74c3c")

    z = np.polyfit(years, win_rates, 3)
    p = np.poly1d(z)
    x_smooth = np.linspace(min(years), max(years), 200)
    ax.plot(x_smooth, p(x_smooth), color="#2c3e50", linewidth=2.5, linestyle="--", alpha=0.7, label="Trend (cubic fit)")

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Decade", fontsize=13)
    ax.set_ylabel("Win Rate", fontsize=13)
    ax.set_title("LitBench: Win Rate by Decade\n(Claude Opus 4.6 head-to-head literary comparisons)", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(min(years) - 15, max(years) + 15)
    ax.set_xticks(years[::2])
    ax.set_xticklabels([f"{y}s" for y in years[::2]], rotation=45, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    def first_book_label(entry):
        b = entry if isinstance(entry, str) else entry[0] if entry else ""
        return b.replace("*", "").split("—")[0].split("by")[0].strip()[:25]

    rated = sorted(zip(decades, years, win_rates), key=lambda x: x[2], reverse=True)
    for d, y, r in rated[:3]:
        label = first_book_label(books.get(d, ""))
        ax.annotate(label, (y, r), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7, color="#27ae60", fontweight="bold")
    for d, y, r in rated[-3:]:
        label = first_book_label(books.get(d, ""))
        ax.annotate(label, (y, r), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7, color="#c0392b", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    click.echo(f"Plot saved to {out_path}")


@click.command()
@click.option("--rounds", default=3, help="Resampling rounds")
@click.option("--max-concurrent", default=10, help="Max concurrent API calls")
@click.option("--start-decade", default=1590, type=int)
@click.option("--end-decade", default=2020, type=int)
@click.option(
    "--comparisons",
    default=None,
    type=int,
    help="Max comparisons per round (default: all pairs)",
)
@click.option("--output", default=None, type=click.Path(), help="Output JSON path")
def main(rounds, max_concurrent, start_decade, end_decade, comparisons, output):
    decades = [f"{y}s" for y in range(start_decade, end_decade + 1, 10)]
    n_pairs = len(decades) * (len(decades) - 1) // 2
    comps_per_round = min(comparisons, n_pairs) if comparisons else n_pairs
    n_calls = rounds * (len(decades) + comps_per_round)

    click.echo(f"LitBench: {len(decades)} decades, {rounds} rounds")
    click.echo(f"Model: {MODEL} (adaptive thinking, effort=high)")
    click.echo(
        f"API calls: ~{n_calls} ({len(decades)} nominations + {comps_per_round} comparisons per round)"
    )

    client = AsyncAnthropic(max_retries=5)
    sem = asyncio.Semaphore(max_concurrent)

    async def run():
        results = []
        for i in range(1, rounds + 1):
            result = await run_round(client, sem, decades, i, comparisons)
            results.append(result)
        return results

    all_rounds = asyncio.run(run())
    print_summary(all_rounds, decades)

    out_path = (
        Path(output)
        if output
        else RESULTS_DIR / f"litbench_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_rounds, f, indent=2)
    click.echo(f"\nResults saved to {out_path}")

    plot_path = out_path.with_suffix(".png")
    plot_results(all_rounds, decades, plot_path)


if __name__ == "__main__":
    main()
