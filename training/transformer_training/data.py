import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, TextIO, Tuple

import torch
from torch.utils.data import Dataset

RISK_COUNTRIES = {"RU", "IR", "KP", "AF", "SY"}
MAX_TRANSACTIONS_DEFAULT = 20
TX_FEATURE_DIM = 5  # amount, hour_norm, is_night, is_wire, is_risk_country


def _open_dataset_file(path: str) -> TextIO:
    if path.startswith("gs://"):
        try:
            import gcsfs  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "gcsfs is required to read datasets from Google Cloud Storage."
            ) from exc

        fs = gcsfs.GCSFileSystem()
        return fs.open(path, "r", encoding="utf-8")
    return Path(path).open("r", encoding="utf-8")


def _log_norm(value: float) -> float:
    return math.log1p(max(value, 0.0))


def _parse_timestamp(ts: str) -> datetime | None:
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _build_vocab(values: Sequence[str]) -> Tuple[List[str], Dict[str, int]]:
    vocab = ["<unk>"] + sorted(set(values))
    mapping = {token: idx for idx, token in enumerate(vocab)}
    return vocab, mapping


def _ratio(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def encode_facts(
    facts: dict,
    *,
    max_transactions: int,
    segment_to_id: Dict[str, int],
    home_country_to_id: Dict[str, int],
    channel_to_id: Dict[str, int],
    counterparty_country_to_id: Dict[str, int],
) -> Dict[str, torch.Tensor | int | List[float]]:
    person = facts.get("person", {})
    account = facts.get("account", {})
    txs = facts.get("recent_tx", []) or []

    segment_idx = segment_to_id.get(str(person.get("segment", "")).lower(), 0)
    home_country = str(person.get("home_country", "")).upper()
    home_country_idx = home_country_to_id.get(home_country, 0)

    pep = 1.0 if person.get("pep") else 0.0
    sanctions = 1.0 if person.get("sanctions_hit") else 0.0
    kyc = 1.0 if person.get("kyc_verified") else 0.0
    avg_amount = _log_norm(float(person.get("avg_tx_amount_90d", 0.0)))
    avg_per_day = float(person.get("avg_tx_per_day_90d", 0.0))
    opened_days = _log_norm(float(account.get("opened_days_ago", 0.0)))

    tx_count = min(len(txs), max_transactions)
    amounts = [float(tx.get("amount", 0.0)) for tx in txs[:tx_count]]
    total_amount = _log_norm(sum(amounts))
    avg_recent = _log_norm(sum(amounts) / tx_count) if tx_count else 0.0
    max_recent = _log_norm(max(amounts)) if amounts else 0.0
    min_recent = _log_norm(min(amounts)) if amounts else 0.0

    out_count = 0
    wire_count = 0
    risk_count = 0

    tx_numeric = torch.zeros(max_transactions, TX_FEATURE_DIM, dtype=torch.float32)
    tx_channel_idx = torch.zeros(max_transactions, dtype=torch.long)
    tx_country_idx = torch.zeros(max_transactions, dtype=torch.long)
    tx_direction_idx = torch.zeros(max_transactions, dtype=torch.long)
    tx_mask = torch.zeros(max_transactions, dtype=torch.bool)

    for i, tx in enumerate(txs[:max_transactions]):
        amount = _log_norm(float(tx.get("amount", 0.0)))
        timestamp = tx.get("timestamp")
        dt = _parse_timestamp(timestamp) if timestamp else None
        hour = dt.hour if dt else 0
        hour_norm = hour / 23.0
        is_night = 1.0 if hour < 6 or hour >= 22 else 0.0

        channel = str(tx.get("channel", "")).lower()
        channel_idx = channel_to_id.get(channel, 0)
        is_wire = 1.0 if channel == "wire" else 0.0

        direction = str(tx.get("direction", "")).lower()
        direction_idx = 1 if direction == "out" else 0
        if direction_idx == 1:
            out_count += 1

        counterparty_country = str(tx.get("counterparty_country", "")).upper()
        country_idx = counterparty_country_to_id.get(counterparty_country, 0)
        is_risk_country = 1.0 if counterparty_country in RISK_COUNTRIES else 0.0
        if is_wire:
            wire_count += 1
        if is_risk_country:
            risk_count += 1

        tx_numeric[i] = torch.tensor(
            [amount, hour_norm, is_night, is_wire, is_risk_country],
            dtype=torch.float32,
        )
        tx_channel_idx[i] = channel_idx
        tx_country_idx[i] = country_idx
        tx_direction_idx[i] = direction_idx
        tx_mask[i] = True

    global_numeric = [
        pep,
        sanctions,
        kyc,
        avg_amount,
        avg_per_day,
        opened_days,
        float(tx_count),
        total_amount,
        avg_recent,
        max_recent,
        min_recent,
        _ratio(out_count, tx_count),
        _ratio(wire_count, tx_count),
        _ratio(risk_count, tx_count),
    ]

    return {
        "global_numeric": torch.tensor(global_numeric, dtype=torch.float32),
        "segment_idx": segment_idx,
        "home_country_idx": home_country_idx,
        "tx_numeric": tx_numeric,
        "tx_channel_idx": tx_channel_idx,
        "tx_country_idx": tx_country_idx,
        "tx_direction_idx": tx_direction_idx,
        "tx_mask": tx_mask,
    }


class AMLTransformerDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        max_transactions: int = MAX_TRANSACTIONS_DEFAULT,
    ) -> None:
        self.dataset_path = dataset_path
        self.max_transactions = max_transactions
        (
            self.reason_vocab,
            segment_values,
            home_country_values,
            channel_values,
            counterparty_country_values,
        ) = self._scan_metadata()

        (
            self.segment_list,
            self.segment_to_id,
        ) = _build_vocab(segment_values)
        (
            self.home_country_list,
            self.home_country_to_id,
        ) = _build_vocab(home_country_values)
        (
            self.channel_list,
            self.channel_to_id,
        ) = _build_vocab(channel_values)
        (
            self.counterparty_country_list,
            self.counterparty_country_to_id,
        ) = _build_vocab(counterparty_country_values)

        self.samples = self._load_samples()
        self.global_feature_dim = len(
            self.samples[0]["global_numeric"]
        ) if self.samples else len(
            encode_facts(
                {"person": {}, "account": {}, "recent_tx": []},
                max_transactions=self.max_transactions,
                segment_to_id=self.segment_to_id,
                home_country_to_id=self.home_country_to_id,
                channel_to_id=self.channel_to_id,
                counterparty_country_to_id=self.counterparty_country_to_id,
            )["global_numeric"]
        )
        self.tx_feature_dim = TX_FEATURE_DIM

    def _scan_metadata(self):
        reason_set = set()
        segments = set()
        home_countries = set()
        channels = set()
        counterparty_countries = set()

        with _open_dataset_file(self.dataset_path) as fh:
            for line in fh:
                record = json.loads(line)
                decision = record.get("decision", {})
                for reason in decision.get("reasons", []):
                    reason_set.add(reason)
                facts = record.get("facts", {})
                person = facts.get("person", {})
                segments.add(str(person.get("segment", "")).lower())
                home_countries.add(str(person.get("home_country", "")).upper())
                for tx in facts.get("recent_tx", []) or []:
                    channels.add(str(tx.get("channel", "")).lower())
                    counterparty_countries.add(
                        str(tx.get("counterparty_country", "")).upper()
                    )

        if not reason_set:
            raise ValueError(
                "No reasons were found in the dataset. Cannot train the reason head."
            )
        return (
            sorted(reason_set),
            segments,
            home_countries,
            channels,
            counterparty_countries,
        )

    def _load_samples(self):
        samples = []
        with _open_dataset_file(self.dataset_path) as fh:
            for line in fh:
                record = json.loads(line)
                facts = record["facts"]
                features = encode_facts(
                    facts,
                    max_transactions=self.max_transactions,
                    segment_to_id=self.segment_to_id,
                    home_country_to_id=self.home_country_to_id,
                    channel_to_id=self.channel_to_id,
                    counterparty_country_to_id=self.counterparty_country_to_id,
                )
                decision = record["decision"]["aml_decision"]
                escalation = record["decision"].get("escalation_level", 0)
                reasons = record["decision"].get("reasons", [])
                reason_vec = [0.0] * len(self.reason_vocab)
                reason_index = {name: idx for idx, name in enumerate(self.reason_vocab)}
                for reason in reasons:
                    idx = reason_index.get(reason)
                    if idx is not None:
                        reason_vec[idx] = 1.0

                sample = {
                    **features,
                    "decision": DECISION_TO_ID[decision],
                    "escalation": min(max(0, escalation), MAX_ESCALATION),
                    "reason_vec": torch.tensor(reason_vec, dtype=torch.float32),
                }
                samples.append(sample)

        if not samples:
            raise ValueError(f"Dataset {self.dataset_path} is empty.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        return sample

    @property
    def feature_vocabs(self) -> Dict[str, List[str]]:
        return {
            "segment": self.segment_list,
            "home_country": self.home_country_list,
            "channel": self.channel_list,
            "counterparty_country": self.counterparty_country_list,
        }

    @property
    def segment_vocab_size(self) -> int:
        return len(self.segment_list)

    @property
    def home_country_vocab_size(self) -> int:
        return len(self.home_country_list)

    @property
    def channel_vocab_size(self) -> int:
        return len(self.channel_list)

    @property
    def counterparty_country_vocab_size(self) -> int:
        return len(self.counterparty_country_list)


def collate_batch(batch: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    global_numeric = torch.stack([item["global_numeric"] for item in batch])
    segment_idx = torch.tensor([item["segment_idx"] for item in batch], dtype=torch.long)
    home_country_idx = torch.tensor(
        [item["home_country_idx"] for item in batch], dtype=torch.long
    )
    tx_numeric = torch.stack([item["tx_numeric"] for item in batch])
    tx_channel_idx = torch.stack([item["tx_channel_idx"] for item in batch])
    tx_country_idx = torch.stack([item["tx_country_idx"] for item in batch])
    tx_direction_idx = torch.stack([item["tx_direction_idx"] for item in batch])
    tx_mask = torch.stack([item["tx_mask"] for item in batch])
    decision_labels = torch.tensor([item["decision"] for item in batch], dtype=torch.long)
    escalation_labels = torch.tensor(
        [item["escalation"] for item in batch], dtype=torch.long
    )
    reason_labels = torch.stack([item["reason_vec"] for item in batch])

    return {
        "global_numeric": global_numeric,
        "segment_idx": segment_idx,
        "home_country_idx": home_country_idx,
        "tx_numeric": tx_numeric,
        "tx_channel_idx": tx_channel_idx,
        "tx_country_idx": tx_country_idx,
        "tx_direction_idx": tx_direction_idx,
        "tx_mask": tx_mask,
        "decision_labels": decision_labels,
        "escalation_labels": escalation_labels,
        "reason_labels": reason_labels,
    }


DECISION_TO_ID = {"CLEAR": 0, "REVIEW": 1, "SAR": 2, "BLOCK": 3}
ID_TO_DECISION = {v: k for k, v in DECISION_TO_ID.items()}
MAX_ESCALATION = 3
