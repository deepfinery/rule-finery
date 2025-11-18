#!/usr/bin/env python3
"""Benchmark the deployed AML LoRA endpoint using the training dataset."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import requests

DEFAULT_SYSTEM_PROMPT = (
    "You are an AML rule engine simulator. "
    "Return ONLY JSON with keys aml_decision, reasons, escalation_level."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data-gen/dataset/tx_aml_dataset.jsonl",
        help="Path to the JSONL dataset used for training.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the vLLM OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--model",
        default="aml-qlora",
        help="Model name to send in the OpenAI request payload.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of records to evaluate.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N rows of the dataset (useful for sharding).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Generation max tokens per request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature used in the request payload.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="HTTP timeout per request in seconds.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Override the system prompt sent to the model.",
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path to write the aggregate metrics as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print model responses and errors for each record.",
    )
    return parser.parse_args()


def iter_dataset(path: Path, skip: int = 0, limit: int | None = None) -> Iterable[Dict[str, Any]]:
    count = 0
    emitted = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            if count <= skip:
                continue
            yield json.loads(line)
            emitted += 1
            if limit is not None and emitted >= limit:
                break


def extract_json_blob(text: str) -> Dict[str, Any]:
    """Return the first JSON object found in the model response."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model response does not contain JSON.")
    snippet = text[start : end + 1]
    return json.loads(snippet)


def reason_counts(pred: Iterable[str], gold: Iterable[str]) -> Tuple[int, int, int]:
    pred_set = {r.strip().upper() for r in pred if r}
    gold_set = {r.strip().upper() for r in gold if r}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def run_benchmark(
    dataset_path: Path,
    endpoint: str,
    model: str,
    *,
    system_prompt: str,
    limit: int | None,
    skip: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
    verbose: bool = False,
) -> Dict[str, Any]:
    stats = {
        "count": 0,
        "decision_correct": 0,
        "escalation_correct": 0,
        "exact_match": 0,
        "reasons_tp": 0,
        "reasons_fp": 0,
        "reasons_fn": 0,
        "latencies": [],
        "failures": 0,
    }

    start_time = time.perf_counter()
    for row in iter_dataset(dataset_path, skip=skip, limit=limit):
        facts = row.get("facts", {})
        gold = row.get("decision", {})

        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Facts:\n{json.dumps(facts, ensure_ascii=False)}",
                },
            ],
        }

        stats["count"] += 1
        sent_at = time.perf_counter()
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout)
            latency = time.perf_counter() - sent_at
            stats["latencies"].append(latency)
            if response.status_code != 200:
                stats["failures"] += 1
                if verbose:
                    print(f"[ERROR] HTTP {response.status_code}: {response.text}", file=sys.stderr)
                continue

            body = response.json()
            choice = body["choices"][0]["message"]["content"]
            parsed = extract_json_blob(choice)
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            stats["failures"] += 1
            if verbose:
                print(f"[ERROR] {exc}", file=sys.stderr)
            continue

        aml_decision = str(parsed.get("aml_decision", "")).upper()
        gold_decision = str(gold.get("aml_decision", "")).upper()
        escalation = int(parsed.get("escalation_level", -1))
        gold_escalation = int(gold.get("escalation_level", -1))
        reasons = parsed.get("reasons", [])
        if isinstance(reasons, str):
            reasons = [reasons]

        stats["decision_correct"] += int(aml_decision == gold_decision)
        stats["escalation_correct"] += int(escalation == gold_escalation)
        tp, fp, fn = reason_counts(reasons, gold.get("reasons", []))
        stats["reasons_tp"] += tp
        stats["reasons_fp"] += fp
        stats["reasons_fn"] += fn

        if (aml_decision == gold_decision) and (escalation == gold_escalation) and (fp == 0 and fn == 0):
            stats["exact_match"] += 1

        if verbose:
            print(
                f"[{stats['count']}] {aml_decision}/{escalation} | "
                f"gold={gold_decision}/{gold_escalation} | latency={latency:.2f}s"
            )

    duration = time.perf_counter() - start_time
    evaluated = stats["count"] - stats["failures"]
    if evaluated == 0:
        raise RuntimeError("No successful responses to evaluate.")

    decision_acc = stats["decision_correct"] / evaluated
    escalation_acc = stats["escalation_correct"] / evaluated
    exact_match = stats["exact_match"] / evaluated
    prec = stats["reasons_tp"] / max(stats["reasons_tp"] + stats["reasons_fp"], 1)
    rec = stats["reasons_tp"] / max(stats["reasons_tp"] + stats["reasons_fn"], 1)
    reason_f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    median_latency = statistics.median(stats["latencies"]) if stats["latencies"] else 0.0
    avg_latency = statistics.fmean(stats["latencies"]) if stats["latencies"] else 0.0
    throughput = evaluated / duration if duration > 0 else 0.0

    return {
        "dataset": str(dataset_path),
        "endpoint": endpoint,
        "model": model,
        "total": stats["count"],
        "evaluated": evaluated,
        "failures": stats["failures"],
        "decision_accuracy": decision_acc,
        "escalation_accuracy": escalation_acc,
        "exact_match": exact_match,
        "reasons_precision": prec,
        "reasons_recall": rec,
        "reasons_f1": reason_f1,
        "avg_latency": avg_latency,
        "median_latency": median_latency,
        "throughput": throughput,
        "duration_seconds": duration,
    }


def print_summary(metrics: Dict[str, Any]) -> None:
    print("\n=== Benchmark Summary ===")
    print(f"Dataset:            {metrics['dataset']}")
    print(f"Endpoint:           {metrics['endpoint']}")
    print(f"Model:              {metrics['model']}")
    print(f"Records evaluated:  {metrics['evaluated']} / {metrics['total']}")
    print(f"Failures:           {metrics['failures']}")
    print(f"Decision accuracy:  {metrics['decision_accuracy']:.3f}")
    print(f"Escalation accuracy:{metrics['escalation_accuracy']:.3f}")
    print(f"Exact match:        {metrics['exact_match']:.3f}")
    print(f"Reasons precision:  {metrics['reasons_precision']:.3f}")
    print(f"Reasons recall:     {metrics['reasons_recall']:.3f}")
    print(f"Reasons F1:         {metrics['reasons_f1']:.3f}")
    print(f"Avg latency (s):    {metrics['avg_latency']:.2f}")
    print(f"Median latency (s): {metrics['median_latency']:.2f}")
    print(f"Throughput (req/s): {metrics['throughput']:.2f}")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        sys.exit(f"Dataset not found: {dataset_path}")
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"

    try:
        metrics = run_benchmark(
            dataset_path,
            endpoint,
            args.model,
            system_prompt=args.system_prompt,
            limit=args.limit,
            skip=args.skip,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            verbose=args.verbose,
        )
    except RuntimeError as exc:
        sys.exit(str(exc))

    print_summary(metrics)

    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
