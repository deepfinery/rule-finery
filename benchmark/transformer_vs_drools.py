#!/usr/bin/env python3
"""Benchmark transformer vs Drools on freshly generated cases."""

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.transformer_training.data import encode_facts, ID_TO_DECISION
from training.transformer_training.model import MultiHeadAMLTransformer, TransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare transformer vs Drools.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of random cases.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="training/transformer_training/artifacts/transformer_multihead.pt",
        help="Transformer checkpoint path.",
    )
    parser.add_argument(
        "--drools-jar",
        type=str,
        default="rule-engine/drool-runner/target/drools-runner-1.0.0-shaded.jar",
    )
    parser.add_argument(
        "--drools-rules",
        type=str,
        default="rule-engine/rules/tx_aml.drl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reason-threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-json",
        type=str,
        default="benchmark/transformer_vs_drools.json",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_transformer(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = TransformerConfig(**ckpt["config"])
    model = MultiHeadAMLTransformer(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    aux = {
        "reason_vocab": ckpt["reason_vocab"],
        "feature_vocabs": ckpt["feature_vocabs"],
        "max_transactions": ckpt["max_transactions"],
    }
    return model, aux


def load_drools_module(repo_root: Path):
    make_path = repo_root / "data-gen" / "make_tx_aml_dataset.py"
    spec = importlib.util.spec_from_file_location("make_tx", make_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def prepare_transformer_inputs(
    facts: dict,
    *,
    max_transactions: int,
    segment_to_id: dict,
    home_country_to_id: dict,
    channel_to_id: dict,
    counterparty_country_to_id: dict,
    device: torch.device,
):
    features = encode_facts(
        facts,
        max_transactions=max_transactions,
        segment_to_id=segment_to_id,
        home_country_to_id=home_country_to_id,
        channel_to_id=channel_to_id,
        counterparty_country_to_id=counterparty_country_to_id,
    )
    inputs = {
        "global_numeric": features["global_numeric"].unsqueeze(0).to(device),
        "segment_idx": torch.tensor([features["segment_idx"]], dtype=torch.long, device=device),
        "home_country_idx": torch.tensor([features["home_country_idx"]], dtype=torch.long, device=device),
        "tx_numeric": features["tx_numeric"].unsqueeze(0).to(device),
        "tx_channel_idx": features["tx_channel_idx"].unsqueeze(0).to(device),
        "tx_country_idx": features["tx_country_idx"].unsqueeze(0).to(device),
        "tx_direction_idx": features["tx_direction_idx"].unsqueeze(0).to(device),
        "tx_mask": features["tx_mask"].unsqueeze(0).to(device),
    }
    return inputs


def benchmark(samples: int, args: argparse.Namespace, repo_root: Path) -> dict:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    model, aux = load_transformer(checkpoint_path, device)
    reason_vocab = aux["reason_vocab"]
    feature_vocabs = aux["feature_vocabs"]
    mappings = {
        "segment_to_id": {token: idx for idx, token in enumerate(feature_vocabs["segment"])},
        "home_country_to_id": {
            token: idx for idx, token in enumerate(feature_vocabs["home_country"])
        },
        "channel_to_id": {token: idx for idx, token in enumerate(feature_vocabs["channel"])},
        "counterparty_country_to_id": {
            token: idx for idx, token in enumerate(feature_vocabs["counterparty_country"])
        },
    }

    drools = load_drools_module(repo_root)
    drools.JAR = Path(args.drools_jar)
    drools.DRL = Path(args.drools_rules)

    rng = random.Random(args.seed)

    drools_total = 0.0
    transformer_total = 0.0
    correct_decisions = 0
    correct_escalations = 0
    tp = fp = fn = 0.0

    for _ in tqdm(range(samples), desc="Comparing"):
        case = drools.make_case(rng)
        start = time.perf_counter()
        drools_out = drools.eval_case(case)
        drools_total += time.perf_counter() - start
        training_rec, _ = drools.clean_labeled(drools_out)
        truth_decision = training_rec["decision"]["aml_decision"]
        truth_escalation = training_rec["decision"]["escalation_level"]
        truth_reasons = set(training_rec["decision"]["reasons"])

        inputs = prepare_transformer_inputs(
            training_rec["facts"],
            max_transactions=aux["max_transactions"],
            device=device,
            **mappings,
        )
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        transformer_total += time.perf_counter() - start

        pred_decision = ID_TO_DECISION[outputs["decision_logits"].argmax(dim=-1).item()]
        pred_escalation = int(outputs["escalation_logits"].argmax(dim=-1).item())

        if pred_decision == truth_decision:
            correct_decisions += 1
        if pred_escalation == truth_escalation:
            correct_escalations += 1

        scores = torch.sigmoid(outputs["reason_logits"]).squeeze(0).tolist()
        pred_reasons = {
            reason_vocab[i]
            for i, score in enumerate(scores)
            if score >= args.reason_threshold
        }
        tp += len(pred_reasons & truth_reasons)
        fp += len(pred_reasons - truth_reasons)
        fn += len(truth_reasons - pred_reasons)

    drools_avg = drools_total / samples if samples else 0.0
    transformer_avg = transformer_total / samples if samples else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "samples": samples,
        "drools_avg_latency_sec": drools_avg,
        "drools_throughput_per_sec": samples / drools_total if drools_total else 0.0,
        "transformer_avg_latency_sec": transformer_avg,
        "transformer_throughput_per_sec": samples / transformer_total if transformer_total else 0.0,
        "decision_accuracy": correct_decisions / samples if samples else 0.0,
        "escalation_accuracy": correct_escalations / samples if samples else 0.0,
        "reason_precision": precision,
        "reason_recall": recall,
        "reason_f1": f1,
    }


def main():
    args = parse_args()
    metrics = benchmark(args.samples, args, REPO_ROOT)
    output = {
        "checkpoint": args.checkpoint,
        "drools_jar": args.drools_jar,
        "samples": args.samples,
        "metrics": metrics,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
