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


async def nominate(client, sem, decade):
    return await ask(
        client,
        sem,
        f"Name the single best piece of literature from the {decade}. "
        "Reply with just the title and author, nothing else.",
    )


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
    nominations = dict(zip(decades, nom_results))

    for d in decades:
        click.echo(f"  {d}: {nominations[d]}")

    all_pairs = list(combinations(decades, 2))
    if max_comparisons and max_comparisons < len(all_pairs):
        pairs = random.sample(all_pairs, max_comparisons)
    else:
        pairs = all_pairs
    click.echo(f"\nComparing {len(pairs)} pairs...")

    comparisons = await asyncio.gather(
        *[compare(client, sem, nominations[a], a, nominations[b], b) for a, b in pairs]
    )

    decided = sum(1 for c in comparisons if c["winner"])
    click.echo(f"  {decided}/{len(pairs)} decided")

    return {"nominations": nominations, "comparisons": comparisons}


def print_summary(all_rounds, decades):
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

    client = AsyncAnthropic()
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


if __name__ == "__main__":
    main()
