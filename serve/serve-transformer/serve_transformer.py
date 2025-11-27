#!/usr/bin/env python3
"""FastAPI server that hosts the structured AML transformer."""

import argparse
import json
from pathlib import Path
from typing import List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from training.transformer_training.data import (
    encode_facts,
    ID_TO_DECISION,
)
from training.transformer_training.model import (
    MultiHeadAMLTransformer,
    TransformerConfig,
)


class PredictRequest(BaseModel):
    facts: dict = Field(..., description="Structured facts object.")
    top_k_reasons: int = Field(3, ge=1, le=50)


class PredictResponse(BaseModel):
    aml_decision: str
    escalation_level: int
    reasons: List[str]
    reason_scores: List[float]


class TransformerService:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.config = TransformerConfig(**ckpt["config"])
        self.reason_vocab = ckpt["reason_vocab"]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_transactions = ckpt.get("max_transactions", self.config.max_transactions)
        feature_vocabs = ckpt["feature_vocabs"]
        self.segment_to_id = {token: idx for idx, token in enumerate(feature_vocabs["segment"])}
        self.home_country_to_id = {
            token: idx for idx, token in enumerate(feature_vocabs["home_country"])
        }
        self.channel_to_id = {token: idx for idx, token in enumerate(feature_vocabs["channel"])}
        self.counterparty_country_to_id = {
            token: idx for idx, token in enumerate(feature_vocabs["counterparty_country"])
        }

        self.model = MultiHeadAMLTransformer(self.config)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self, facts: dict, top_k_reasons: int = 3
    ) -> PredictResponse:
        features = encode_facts(
            facts,
            max_transactions=self.max_transactions,
            segment_to_id=self.segment_to_id,
            home_country_to_id=self.home_country_to_id,
            channel_to_id=self.channel_to_id,
            counterparty_country_to_id=self.counterparty_country_to_id,
        )
        inputs = {
            "global_numeric": features["global_numeric"].unsqueeze(0).to(self.device),
            "segment_idx": torch.tensor(
                [features["segment_idx"]], dtype=torch.long, device=self.device
            ),
            "home_country_idx": torch.tensor(
                [features["home_country_idx"]], dtype=torch.long, device=self.device
            ),
            "tx_numeric": features["tx_numeric"].unsqueeze(0).to(self.device),
            "tx_channel_idx": features["tx_channel_idx"].unsqueeze(0).to(self.device),
            "tx_country_idx": features["tx_country_idx"].unsqueeze(0).to(self.device),
            "tx_direction_idx": features["tx_direction_idx"].unsqueeze(0).to(
                self.device
            ),
            "tx_mask": features["tx_mask"].unsqueeze(0).to(self.device),
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            decision_id = outputs["decision_logits"].argmax(dim=-1).item()
            escalation_level = outputs["escalation_logits"].argmax(dim=-1).item()
            reason_scores = torch.sigmoid(outputs["reason_logits"]).squeeze(0).tolist()

        decision = ID_TO_DECISION.get(decision_id, "UNKNOWN")
        ranked = sorted(
            zip(self.reason_vocab, reason_scores),
            key=lambda item: item[1],
            reverse=True,
        )
        reasons, scores = zip(
            *[(name, score) for name, score in ranked[:top_k_reasons]]
        ) if ranked else ([], [])

        return PredictResponse(
            aml_decision=decision,
            escalation_level=int(escalation_level),
            reasons=list(reasons),
            reason_scores=[float(s) for s in scores],
        )


def create_app(service: TransformerService) -> FastAPI:
    app = FastAPI(title="AML Transformer Service")

    @app.get("/healthz")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        return service.predict(
            request.facts,
            top_k_reasons=request.top_k_reasons,
        )

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Serve the transformer model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="training/transformer_training/artifacts/transformer_multihead.pt",
        help="Path to the saved checkpoint.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--log-level", type=str, default="info", help="uvicorn log level."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    service = TransformerService(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
