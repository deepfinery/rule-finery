#!/usr/bin/env python3
# make_tx_aml_dataset_v3_balanced.py
import os, sys, json, math, random, pathlib, subprocess
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool, cpu_count
from collections import Counter

# --- Paths ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
JAR  = ROOT / "rule-engine" / "drool-runner" / "target" / "drools-runner-1.0.0-shaded.jar"
DRL  = ROOT / "rule-engine" / "rules" / "tx_aml.drl"

# --- Countries & scenarios ---
SAFE = ["CA","US","GB","FR","DE","MX","IN","AE","ES","IT","SE","NL","JP","AU"]
RISK = ["RU","IR","KP","AF","SY"]

# Scenario prior (used to sample *facts*); final class is enforced by quotas after labeling
SCENARIO_PROBS = {
    "normal":            0.40,
    "behavior_shift":    0.15,
    "structuring":       0.12,
    "large_wire":        0.10,
    "round_trip":        0.08,
    "odd_hour":          0.05,
    "pep_hit":           0.06,
    "sanctions_hit":     0.04,  # NEW: improves BLOCK coverage
}

# --- Target class ratios (overall) ---
TARGET_RATIOS = {
    "CLEAR":  0.48,
    "REVIEW": 0.30,
    "SAR":    0.15,
    "BLOCK":  0.07,
}

# Canonical escalation mapping
CANON_ESC = {"CLEAR": 0, "REVIEW": 1, "SAR": 2, "BLOCK": 3}

# Fired-rules boilerplate to drop in training view (kept in audit)
NOISE_RULES = {"INIT", "FINAL_DECISION"}

# ---------------------------------------------------------------------------

def pick_scenario(rng):
    r = rng.random(); s = 0.0
    for k, p in SCENARIO_PROBS.items():
        s += p
        if r < s:
            return k
    return "normal"

def rand_ts(rng, days_back=3, night=False):
    base = datetime.now(timezone.utc) - timedelta(days=rng.uniform(0, days_back))
    hour = rng.choice([0,1,2,3,4]) if night else rng.choice([9,10,11,12,13,14,15,16,17])
    t = base.replace(hour=hour, minute=rng.randint(0,59), second=rng.randint(0,59), microsecond=0)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")

def clip(x, lo, hi): return max(lo, min(hi, x))

def random_poisson(rng, lam):
    # Knuth algorithm, no numpy needed
    L = math.exp(-lam); k = 0; p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1

def sample_profile(rng, force_kyc=None):
    avg_amt = rng.uniform(25, 400)
    avg_per_day = rng.uniform(0.3, 4.5)
    kyc = (rng.random() > 0.02) if force_kyc is None else force_kyc
    return dict(
        pep=False,
        sanctions_hit=False,
        home_country=rng.choice(SAFE),
        kyc_verified=kyc,
        avg_tx_amount_90d=round(avg_amt, 2),
        avg_tx_per_day_90d=round(avg_per_day, 2),
    )

def gen_normal_txs(rng, avg_amt, avg_per_day):
    window_days = rng.uniform(0.5, 2.0)
    lam = max(0.2, avg_per_day * window_days)
    n = max(1, int(random_poisson(rng, lam)))
    txs = []
    for _ in range(n):
        amt = rng.gauss(avg_amt, 0.30 * avg_amt)
        amt = round(clip(amt, 1.0, 8000.0), 2)
        txs.append(dict(
            tx_id=f"T{rng.randint(100000,999999)}",
            timestamp=rand_ts(rng, days_back=3, night=False),
            amount=amt, currency="CAD",
            direction=rng.choice(["in","out"]),
            channel=rng.choice(["online","branch","atm"]),
            counterparty_country=rng.choice(SAFE),
            counterparty_id=f"CP{rng.randint(100,999)}",
            merchant_category=None,
        ))
    return txs

def gen_behavior_shift_txs(rng, avg_amt, avg_per_day):
    window_days = rng.uniform(0.5, 2.0)
    if rng.random() < 0.5:
        factor = rng.uniform(2.0, 4.0)  # count spike
        lam = max(1.0, avg_per_day * window_days * factor)
        n = int(random_poisson(rng, lam))
        n = max(3, min(n, 20))
        amt_mu = avg_amt
    else:
        factor = rng.uniform(2.0, 4.0)  # amount spike
        lam = max(0.5, avg_per_day * window_days)
        n = max(1, int(random_poisson(rng, lam)))
        amt_mu = avg_amt * factor
    txs = []
    for _ in range(n):
        amt = rng.gauss(amt_mu, 0.25 * amt_mu)
        amt = round(clip(amt, 50.0, 25000.0), 2)
        txs.append(dict(
            tx_id=f"T{rng.randint(100000,999999)}",
            timestamp=rand_ts(rng, days_back=3, night=False),
            amount=amt, currency="CAD",
            direction=rng.choice(["in","out"]),
            channel=rng.choice(["online","branch"]),
            counterparty_country=rng.choice(SAFE + (RISK if rng.random()<0.05 else [])),
            counterparty_id=f"CP{rng.randint(100,999)}",
            merchant_category=None,
        ))
    return txs

def gen_structuring_txs(rng):
    k = rng.randint(3, 6)
    txs = []
    for _ in range(k):
        amt = round(rng.uniform(8200, 9999), 2)
        txs.append(dict(
            tx_id=f"T{rng.randint(100000,999999)}",
            timestamp=rand_ts(rng, days_back=1, night=False),
            amount=amt, currency="CAD",
            direction="in", channel="online",
            counterparty_country="CA", counterparty_id=f"CP{rng.randint(100,999)}",
            merchant_category=None,
        ))
    return txs

def gen_large_wire_txs(rng):
    k = rng.randint(1, 3)
    txs = []
    for _ in range(k):
        amt = round(rng.uniform(10050, 50000), 2)
        txs.append(dict(
            tx_id=f"T{rng.randint(100000,999999)}",
            timestamp=rand_ts(rng, days_back=2, night=False),
            amount=amt, currency="CAD",
            direction=rng.choice(["in","out"]),
            channel="wire",
            counterparty_country=rng.choice(SAFE + (RISK if rng.random()<0.1 else [])),
            counterparty_id=f"CP{rng.randint(100,999)}",
            merchant_category=None,
        ))
    return txs

def gen_round_trip_txs(rng):
    cid = f"CP{rng.randint(100,999)}"
    amt = round(rng.uniform(1500, 7000), 2)
    return [
        dict(tx_id=f"T{rng.randint(100000,999999)}", timestamp=rand_ts(rng, days_back=1, night=False),
             amount=amt, currency="CAD", direction="in", channel="online",
             counterparty_country="CA", counterparty_id=cid, merchant_category=None),
        dict(tx_id=f"T{rng.randint(100000,999999)}", timestamp=rand_ts(rng, days_back=1, night=False),
             amount=round(amt * rng.uniform(0.98,1.02), 2), currency="CAD", direction="out", channel="online",
             counterparty_country="CA", counterparty_id=cid, merchant_category=None),
    ]

def gen_odd_hour_txs(rng, avg_amt, avg_per_day):
    window_days = rng.uniform(0.5, 1.5)
    lam = max(0.5, avg_per_day * window_days)
    n = max(1, int(random_poisson(rng, lam)))
    # To avoid shortcut learning, sometimes generate less suspicious odd-hour activity
    # that might be labeled CLEAR by the rules engine.
    is_benign = rng.random() < 0.40  # Increased from 0.25
    amt_mu = avg_amt * rng.uniform(0.2, 0.8) if is_benign else avg_amt * rng.uniform(1.5, 3.0)
    txs = []
    for _ in range(n):
        amt = round(clip(rng.gauss(amt_mu, 0.25 * amt_mu), 5.0, 6000.0), 2)
        txs.append(dict(
            tx_id=f"T{rng.randint(100000,999999)}",
            timestamp=rand_ts(rng, days_back=1, night=True),
            amount=amt, currency="CAD",
            direction=rng.choice(["in","out"]),
            channel=rng.choice(["online","atm"]),
            counterparty_country="CA",
            counterparty_id=f"CP{rng.randint(100,999)}",
            merchant_category=None,
        ))
    return txs

def make_case(rng):
    # Ensure coverage of KYC-unverified cases across all scenarios.
    force_kyc = False if rng.random() < 0.08 else None

    person_base = sample_profile(rng, force_kyc=force_kyc)
    scenario = pick_scenario(rng)

    pep = person_base["pep"]
    sanctions = person_base["sanctions_hit"]
    if scenario == "pep_hit":
        pep = True
    if scenario == "sanctions_hit":
        sanctions = True
    # rare background sanction regardless of scenario
    if rng.random() < 0.001:
        sanctions = True

    person = dict(
        person_id=f"P{rng.randint(1000,9999)}",
        segment=rng.choice(["retail","premier","student","small_business"]),
        pep=pep,
        sanctions_hit=sanctions,
        home_country=person_base["home_country"],
        kyc_verified=person_base["kyc_verified"],
        avg_tx_amount_90d=person_base["avg_tx_amount_90d"],
        avg_tx_per_day_90d=person_base["avg_tx_per_day_90d"],
    )
    account = dict(account_id=f"A{rng.randint(10000,99999)}", opened_days_ago=rng.randint(30, 2000))

    avg_amt = person["avg_tx_amount_90d"]
    avg_per_day = person["avg_tx_per_day_90d"]

    if scenario == "normal":
        txs = gen_normal_txs(rng, avg_amt, avg_per_day)
    elif scenario == "behavior_shift":
        txs = gen_behavior_shift_txs(rng, avg_amt, avg_per_day)
    elif scenario == "structuring":
        txs = gen_structuring_txs(rng)
    elif scenario == "large_wire":
        txs = gen_large_wire_txs(rng)
    elif scenario == "round_trip":
        txs = gen_round_trip_txs(rng)
    elif scenario == "odd_hour":
        txs = gen_odd_hour_txs(rng, avg_amt, avg_per_day)
    elif scenario in ("pep_hit","sanctions_hit"):
        txs = gen_normal_txs(rng, avg_amt, avg_per_day)
    else:
        txs = gen_normal_txs(rng, avg_amt, avg_per_day)

    return {"person": person, "account": account, "recent_tx": txs}

# --- Drools runner ---
def eval_case(case):
    p = subprocess.Popen(
        ["java","-jar",str(JAR),str(DRL),"-"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = p.communicate(json.dumps(case))
    if p.returncode != 0:
        raise RuntimeError(f"Java failed: {err.strip()[:200]}")
    return json.loads(out)

# --- Cleaning helpers ---
def dedup_list(xs):
    seen = set(); out = []
    for x in xs or []:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def clean_labeled(labeled):
    """
    Accepts the Drools output record (either flat or {facts, decision, fired_rules}).
    Returns:
      training_rec -> cleaned for SFT (dedup reasons, canonical escalation, cleaned fired_rules)
      audit_rec    -> original decision + raw fired_rules preserved
    """
    # Accept both flat and structured formats
    facts    = labeled.get("facts") or {k: labeled.get(k) for k in ("person","account","recent_tx") if k in labeled}
    decision = labeled.get("decision") or {k: labeled.get(k) for k in ("aml_decision","escalation_level","reasons") if k in labeled}
    fired    = labeled.get("fired_rules") or labeled.get("rules") or []

    dec = (decision.get("aml_decision") or "").upper()
    # Use canonical mapping, falling back to original if it exists, otherwise None
    esc = CANON_ESC.get(dec)
    if esc is None:
        esc = decision.get("escalation_level")
    reasons = decision.get("reasons") or []

    # If reasons are duplicated labels (e.g., ["LARGE_WIRE","LARGE_WIRE"]), dedup for training
    if isinstance(reasons, list):
        reasons_clean = dedup_list([str(r) for r in reasons])
    else:
        # already a dict with counts; keep as-is
        reasons_clean = reasons

    # Strip boilerplate + dedup fired rules for training view
    fired_upper = [str(x).upper() for x in (fired or [])]
    fired_clean = [r for r in dedup_list(fired_upper) if r not in NOISE_RULES]

    training = {
        "facts": facts,
        "decision": {
            "aml_decision": dec,
            "escalation_level": esc,
            "reasons": reasons_clean
        },
        "fired_rules": fired_clean
    }
    # keep raw for audit
    audit = {
        "facts": facts,
        "decision": {
            "aml_decision": dec,
            "escalation_level": decision.get("escalation_level"),
            "reasons": reasons
        },
        "fired_rules": fired
    }
    return training, audit

# --- Worker enforcing per-class quotas ---
def _worker(shard_idx, shard_targets, tmp_dir, base_seed, print_every=1000, safety_factor=30):
    """
    shard_targets: dict like {"CLEAR": n1, "REVIEW": n2, "SAR": n3, "BLOCK": n4}
    We generate candidates until we fill the quotas or we hit a safety cap (to avoid infinite loops).
    """
    rng = random.Random(base_seed + shard_idx * 7919)
    out_train = os.path.join(tmp_dir, f"shard_{shard_idx:03d}.jsonl")
    out_audit = os.path.join(tmp_dir, f"shard_{shard_idx:03d}.audit.jsonl")

    counts = Counter()
    need_total = sum(shard_targets.values())
    max_iters  = need_total * safety_factor

    # Warm-up one eval per shard
    _ = eval_case(make_case(rng))

    wrote = 0
    iters = 0
    with open(out_train, "w") as ft, open(out_audit, "w") as fa:
        while wrote < need_total and iters < max_iters:
            iters += 1
            case = make_case(rng)
            labeled = eval_case(case)
            training_rec, audit_rec = clean_labeled(labeled)

            dec = training_rec["decision"]["aml_decision"]
            # only accept if we still need this class
            if counts[dec] < shard_targets.get(dec, 0):
                ft.write(json.dumps(training_rec, separators=(",",":")) + "\n")
                fa.write(json.dumps(audit_rec, separators=(",",":")) + "\n")
                counts[dec] += 1
                wrote += 1
                if wrote % print_every == 0:
                    print(f"[shard {shard_idx}] {wrote}/{need_total}  counts={dict(counts)}")

    if wrote < need_total:
        print(f"[WARN][shard {shard_idx}] quota underfilled {wrote}/{need_total} after {iters} iters. "
              f"Increase scenario odds or safety_factor.")
    return out_train, out_audit, dict(counts)

# --- Main orchestration ---
def main(total=200_000, out_path="dataset/tx_aml_dataset.jsonl", workers=None, seed=42):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_dir = os.path.join(os.path.dirname(out_path) or ".", "_shards_v3")
    os.makedirs(tmp_dir, exist_ok=True)

    if workers is None:
        workers = max(2, min(cpu_count(), 16))

    # Per-class global targets
    global_targets = {k: int(total * v) for k, v in TARGET_RATIOS.items()}
    # Distribute fairly across shards (first shards get the +1 remainder)
    shard_targets = []
    for cls in ("CLEAR","REVIEW","SAR","BLOCK"):
        base = global_targets[cls] // workers
        rem  = global_targets[cls] % workers
        per  = [base + (1 if i < rem else 0) for i in range(workers)]
        for i, val in enumerate(per):
            if len(shard_targets) <= i:
                shard_targets.append({})
            shard_targets[i][cls] = val

    # Launch workers
    args = [(i, shard_targets[i], tmp_dir, seed) for i in range(workers)]
    with Pool(processes=workers) as pool:
        results = pool.starmap(_worker, args)

    # Concatenate shards in order (training + audit)
    out_audit = out_path.replace(".jsonl", ".audit.jsonl")
    total_counts = Counter()
    with open(out_path, "w") as out_tr, open(out_audit, "w") as out_au:
        for i, (train_file, audit_file, cnts) in enumerate(sorted(results, key=lambda x: x[0])):
            total_counts.update(cnts)
            with open(train_file, "r") as f:
                for line in f: out_tr.write(line)
            with open(audit_file, "r") as f:
                for line in f: out_au.write(line)

    # Report
    print(f"\nDONE: wrote training -> {out_path}")
    print(f"      wrote audit    -> {out_audit}")
    tot_written = sum(total_counts.values())
    print(f"Total written: {tot_written} (target {total})  by class: {dict(total_counts)}")
    for cls, want in global_targets.items():
        got = total_counts[cls]
        print(f"  {cls:<6} target={want}  got={got}  delta={got-want}")

if __name__ == "__main__":
    total   = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    outp    = sys.argv[2] if len(sys.argv) > 2 else "dataset/tx_aml_dataset.jsonl"
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else None
    seed    = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    main(total, outp, workers, seed)
