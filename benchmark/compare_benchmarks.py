#!/usr/bin/env python3
"""Compare summary JSON files produced by the benchmarking scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("llm_summary", help="Path to the LLM benchmark summary JSON.")
    parser.add_argument("drools_summary", help="Path to the Drools benchmark summary JSON.")
    return parser.parse_args()


def load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    llm = load(args.llm_summary)
    drools = load(args.drools_summary)

    headers = ["Metric", "LLM", "Drools"]
    rows = [
        ("Decision accuracy", f"{llm['decision_accuracy']:.3f}", f"{drools['decision_accuracy']:.3f}"),
        ("Avg latency (s)", f"{llm['avg_latency']:.2f}", f"{drools['avg_latency']:.2f}"),
        ("Throughput (req/s)", f"{llm['throughput']:.2f}", f"{drools['throughput']:.2f}"),
        ("Exact match", f"{llm.get('exact_match', 0):.3f}", f"{drools.get('exact_match', 0):.3f}"),
    ]

    col_widths = [
        max(len(headers[0]), *(len(r[0]) for r in rows)),
        max(len(headers[1]), *(len(r[1]) for r in rows)),
        max(len(headers[2]), *(len(r[2]) for r in rows)),
    ]

    def fmt_row(values):
        return " | ".join(v.ljust(w) for v, w in zip(values, col_widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


if __name__ == "__main__":
    main()
