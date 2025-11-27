from dataclasses import dataclass

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sine/cosine positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


@dataclass
class TransformerConfig:
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 5
    dim_feedforward: int = 512
    dropout: float = 0.1
    num_decisions: int = 4
    num_escalation_classes: int = 4
    num_reason_labels: int = 1
    global_feature_dim: int = 0
    tx_feature_dim: int = 0
    segment_vocab_size: int = 1
    home_country_vocab_size: int = 1
    channel_vocab_size: int = 1
    counterparty_country_vocab_size: int = 1
    max_transactions: int = 20


class MultiHeadAMLTransformer(nn.Module):
    """Transformer encoder with decision/escalation/reason heads."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.global_linear = nn.Linear(config.global_feature_dim, config.d_model)
        self.segment_emb = nn.Embedding(config.segment_vocab_size, config.d_model)
        self.home_country_emb = nn.Embedding(config.home_country_vocab_size, config.d_model)

        self.tx_linear = nn.Linear(config.tx_feature_dim, config.d_model)
        self.tx_channel_emb = nn.Embedding(config.channel_vocab_size, config.d_model)
        self.tx_country_emb = nn.Embedding(
            config.counterparty_country_vocab_size, config.d_model
        )
        self.tx_direction_emb = nn.Embedding(2, config.d_model)

        self.pos_encoding = PositionalEncoding(config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.dropout = nn.Dropout(config.dropout)

        self.decision_head = nn.Linear(config.d_model, config.num_decisions)
        self.escalation_head = nn.Linear(config.d_model, config.num_escalation_classes)
        self.reason_head = nn.Linear(config.d_model, config.num_reason_labels)

    def forward(
        self,
        global_numeric: torch.Tensor,
        segment_idx: torch.Tensor,
        home_country_idx: torch.Tensor,
        tx_numeric: torch.Tensor,
        tx_channel_idx: torch.Tensor,
        tx_country_idx: torch.Tensor,
        tx_direction_idx: torch.Tensor,
        tx_mask: torch.Tensor,
    ) -> dict:
        global_token = self.global_linear(global_numeric)
        global_token = (
            global_token
            + self.segment_emb(segment_idx)
            + self.home_country_emb(home_country_idx)
        )

        tx_tokens = self.tx_linear(tx_numeric)
        tx_tokens = (
            tx_tokens
            + self.tx_channel_emb(tx_channel_idx)
            + self.tx_country_emb(tx_country_idx)
            + self.tx_direction_emb(tx_direction_idx)
        )

        seq = torch.cat([global_token.unsqueeze(1), tx_tokens], dim=1)
        seq = self.pos_encoding(seq)

        batch_size = tx_mask.size(0)
        cls_mask = torch.ones(
            batch_size, 1, dtype=torch.bool, device=tx_mask.device
        )
        padding_mask = torch.cat([cls_mask, tx_mask.bool()], dim=1)
        encoded = self.encoder(seq, src_key_padding_mask=~padding_mask)

        pooled = self.dropout(encoded[:, 0, :])
        return {
            "decision_logits": self.decision_head(pooled),
            "escalation_logits": self.escalation_head(pooled),
            "reason_logits": self.reason_head(pooled),
        }
