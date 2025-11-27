#!/usr/bin/env python3
"""Train a multi-head transformer on AML structured data."""

import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from .data import AMLTransformerDataset, collate_batch
from .model import MultiHeadAMLTransformer, TransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-head transformer.")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="data-gen/dataset/tx_aml_dataset.jsonl",
        help="JSONL dataset path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/transformer_training/artifacts",
        help="Where to save checkpoints and metadata.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_transactions", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--decision_loss_weight", type=float, default=1.0)
    parser.add_argument("--escalation_loss_weight", type=float, default=0.5)
    parser.add_argument("--reason_loss_weight", type=float, default=0.3)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--upload_output_to",
        type=str,
        default="",
        help="Optional gs:// path to upload the trained artifacts.",
    )
    return parser.parse_args()


def split_dataset(
    dataset, val_ratio: float
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    return random_split(dataset, lengths=[train_size, val_size])


def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct_decisions = 0
    correct_escalations = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = _move_batch_to_device(batch, device)
            inputs.pop("reason_labels", None)
            decision_labels = inputs.pop("decision_labels")
            escalation_labels = inputs.pop("escalation_labels")
            outputs = model(**inputs)
            preds_decision = outputs["decision_logits"].argmax(dim=-1)
            preds_escalation = outputs["escalation_logits"].argmax(dim=-1)
            total += decision_labels.size(0)
            correct_decisions += (preds_decision == decision_labels).sum().item()
            correct_escalations += (preds_escalation == escalation_labels).sum().item()
    return {
        "decision_accuracy": correct_decisions / total,
        "escalation_accuracy": correct_escalations / total,
    }


def _move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def main():
    args = parse_args()
    dataset = AMLTransformerDataset(
        args.dataset_file, max_transactions=args.max_transactions
    )
    train_dataset, val_dataset = split_dataset(dataset, args.val_ratio)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    config = TransformerConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_reason_labels=len(dataset.reason_vocab),
        global_feature_dim=dataset.global_feature_dim,
        tx_feature_dim=dataset.tx_feature_dim,
        segment_vocab_size=dataset.segment_vocab_size,
        home_country_vocab_size=dataset.home_country_vocab_size,
        channel_vocab_size=dataset.channel_vocab_size,
        counterparty_country_vocab_size=dataset.counterparty_country_vocab_size,
        max_transactions=dataset.max_transactions,
    )
    device = torch.device(args.device)
    model = MultiHeadAMLTransformer(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            inputs = _move_batch_to_device(batch, device)
            decision_labels = inputs.pop("decision_labels")
            escalation_labels = inputs.pop("escalation_labels")
            reason_labels = inputs.pop("reason_labels")

            optimizer.zero_grad()
            outputs = model(**inputs)
            decision_loss = F.cross_entropy(outputs["decision_logits"], decision_labels)
            escalation_loss = F.cross_entropy(
                outputs["escalation_logits"], escalation_labels
            )
            reason_loss = F.binary_cross_entropy_with_logits(
                outputs["reason_logits"], reason_labels
            )
            loss = (
                args.decision_loss_weight * decision_loss
                + args.escalation_loss_weight * escalation_loss
                + args.reason_loss_weight * reason_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            progress.set_postfix(
                loss=loss.item(),
                decision=decision_loss.item(),
                escalation=escalation_loss.item(),
                reason=reason_loss.item(),
            )

        metrics = evaluate(model, val_loader, device)
        print(f"Validation metrics: {metrics}")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "reason_vocab": dataset.reason_vocab,
        "feature_vocabs": dataset.feature_vocabs,
        "max_transactions": dataset.max_transactions,
        "args": vars(args),
    }
    ckpt_path = output_dir / "transformer_multihead.pt"
    torch.save(ckpt, ckpt_path)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved metrics to {metrics_path}")

    if args.upload_output_to:
        upload_dir_to_gcs(output_dir, args.upload_output_to)
        print(f"Uploaded artifacts to {args.upload_output_to}")


def upload_dir_to_gcs(local_dir: Path, gcs_uri: str) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("upload_output_to must be a gs:// URI")
    try:
        import gcsfs  # type: ignore
    except ImportError as exc:
        raise ImportError("gcsfs is required to upload artifacts to GCS") from exc

    fs = gcsfs.GCSFileSystem()
    prefix = gcs_uri.rstrip("/")
    for file_path in local_dir.rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(local_dir).as_posix()
        target = f"{prefix}/{rel}"
        with file_path.open("rb") as src, fs.open(target, "wb") as dst:
            dst.write(src.read())


if __name__ == "__main__":
    main()
