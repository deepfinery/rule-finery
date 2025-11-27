#!/usr/bin/env python3
"""Benchmark a trained transformer checkpoint against a dataset."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import AMLTransformerDataset, collate_batch
from .model import MultiHeadAMLTransformer, TransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark transformer checkpoint.")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="data-gen/dataset/tx_aml_dataset.jsonl",
        help="JSONL dataset path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="training/transformer_training/artifacts/transformer_multihead.pt",
        help="Path to checkpoint produced by train.py.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--reason_threshold", type=float, default=0.5)
    parser.add_argument(
        "--output_json",
        type=str,
        default="training/transformer_training/benchmark_metrics.json",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
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
        "args": ckpt.get("args", {}),
    }
    return model, aux


def evaluate(
    model,
    dataloader,
    device,
    reason_threshold: float,
    reason_permute: torch.Tensor,
):
    total = 0
    correct_decisions = 0
    correct_escalations = 0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    total_time = 0.0

    for batch in tqdm(dataloader, desc="Benchmarking"):
        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key
            not in {
                "decision_labels",
                "escalation_labels",
                "reason_labels",
            }
        }
        decision_labels = batch["decision_labels"].to(device)
        escalation_labels = batch["escalation_labels"].to(device)
        reason_labels = batch["reason_labels"][:, reason_permute].to(device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        total_time += time.perf_counter() - start

        preds_decision = outputs["decision_logits"].argmax(dim=-1)
        preds_escalation = outputs["escalation_logits"].argmax(dim=-1)

        total += decision_labels.size(0)
        correct_decisions += (preds_decision == decision_labels).sum().item()
        correct_escalations += (preds_escalation == escalation_labels).sum().item()

        reason_probs = torch.sigmoid(outputs["reason_logits"])
        preds_reason = (reason_probs >= reason_threshold).float()

        tp += (preds_reason * reason_labels).sum().item()
        fp += (preds_reason * (1 - reason_labels)).sum().item()
        fn += ((1 - preds_reason) * reason_labels).sum().item()

    decision_acc = correct_decisions / total
    escalation_acc = correct_escalations / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    avg_latency = total_time / total if total_time else 0.0
    throughput = total / total_time if total_time else 0.0
    return {
        "decision_accuracy": decision_acc,
        "escalation_accuracy": escalation_acc,
        "reason_precision": precision,
        "reason_recall": recall,
        "reason_f1": f1,
        "evaluated": total,
        "avg_latency_sec": avg_latency,
        "throughput_samples_per_sec": throughput,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, aux = load_model(args.checkpoint, device)
    dataset = AMLTransformerDataset(
        args.dataset_file,
        max_transactions=aux["max_transactions"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    # Align reason vocab order
    reason_vocab = aux["reason_vocab"]
    dataset_reason_index = {name: idx for idx, name in enumerate(dataset.reason_vocab)}
    permute_indices = []
    for name in reason_vocab:
        if name not in dataset_reason_index:
            raise ValueError(
                f"Reason '{name}' from checkpoint not found in dataset reason vocab."
            )
        permute_indices.append(dataset_reason_index[name])
    reason_permute = torch.tensor(permute_indices, dtype=torch.long)

    metrics = evaluate(
        model,
        dataloader,
        device,
        args.reason_threshold,
        reason_permute,
    )

    result = {
        "checkpoint": str(checkpoint_path),
        "dataset": args.dataset_file,
        "metrics": metrics,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
