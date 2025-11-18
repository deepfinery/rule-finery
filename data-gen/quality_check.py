#!/usr/bin/env python3
# quality_check.py
import argparse, glob, json, math, os, re, sys
from collections import Counter, defaultdict
from statistics import mean

TARGET_BANDS = {
    "CLEAR":  (0.40, 0.55),
    "REVIEW": (0.25, 0.35),
    "SAR":    (0.10, 0.20),
    "BLOCK":  (0.02, 0.08),
}
CANON_ESC = {"CLEAR": 0, "REVIEW": 1, "SAR": 2, "BLOCK": 3}

# Substring keys used to detect “tough pattern” coverage across reasons/fired_rules
PATTERNS = {
    "STRUCTURING":   ("STRUCTUR",),          # catches STRUCTURING / STRUCTURING_BURST / etc
    "ROUND_TRIP":    ("ROUND_TRIP",),
    "PEP":           ("PEP",),               # PEP_HIT
    "SANCTIONS":     ("SANCTION",),          # SANCTIONS_HIT
    "ODD_HOUR":      ("ODD_HOUR",),
    "KYC_UNVERIFIED":("KYC",),               # KYC_FAILURE
    "LARGE_WIRE":    ("LARGE_WIRE",),        # LARGE_WIRE / LARGE_WIRE_SINGLE
    "BEHAVIOR_SHIFT":("BEHAVIOR_SHIFT",),    # BEHAVIOR_SHIFT_VS_BASELINE
}

def parse_args():
    ap = argparse.ArgumentParser(description="Quality checks for AML distillation dataset")
    ap.add_argument("path", help="JSONL file or a glob of shard files (e.g., dataset/_shards_v3/shard_*.jsonl)")
    ap.add_argument("--max", type=int, default=None, help="Optional max rows to scan")
    ap.add_argument("--dump-json", default=None, help="Optional path to write a JSON summary")
    return ap.parse_args()

def iter_records(files, max_rows=None):
    n = 0
    for fp in files:
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    yield fp, json.loads(line)
                except Exception as e:
                    print(f"[WARN] Bad JSON in {fp}: {e}", file=sys.stderr)
                    continue
                n += 1
                if max_rows and n >= max_rows:
                    return

def get_field(obj, path, default=None):
    cur = obj
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def dedup_list(seq):
    seen = set(); out = []
    for x in seq or []:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def contains_any(texts, needles):
    if not texts: return False
    t = " | ".join(map(str, texts))
    t_up = t.upper()
    return any(n in t_up for n in needles)

def pretty_pct(x):
    return f"{100.0 * x:.2f}%"

def band_status(frac, lo, hi):
    if frac < lo: return "LOW"
    if frac > hi: return "HIGH"
    return "OK"

def main():
    args = parse_args()
    files = sorted(glob.glob(args.path)) if any(ch in args.path for ch in "*?[]") else [args.path]
    if not files:
        print(f"No files matched: {args.path}", file=sys.stderr); sys.exit(2)

    # Aggregates
    total = 0
    by_shard = defaultdict(lambda: Counter())
    class_counter = Counter()
    decision_counts = Counter()
    escalation_counts = Counter()
    decision_escal_pairs = Counter()
    non_clear_empty_reasons = 0
    non_clear_total = 0
    reasons_dup_cases = 0
    reasons_len_stats = []
    reasons_top = Counter()
    fired_rules_noise_cases = 0
    fired_rules_dup_cases = 0
    fired_rules_len_stats = []
    pattern_hits = Counter()
    pattern_decision = defaultdict(lambda: Counter())
    odd_hour_clear = 0
    odd_hour_total = 0
    kyc_false_total = 0
    kyc_false_decision = Counter()

    # Per-shard pattern coverage
    shard_patterns = defaultdict(lambda: Counter())

    # Process records
    for shard, rec in iter_records(files, args.max):
        decision = get_field(rec, "decision.aml_decision") or rec.get("aml_decision")
        esc = get_field(rec, "decision.escalation_level")
        reasons = get_field(rec, "decision.reasons") or rec.get("reasons") or []
        fired = rec.get("fired_rules") or []
        facts = rec.get("facts") or rec  # accept both {facts, decision} or flat
        person = get_field(facts, "person") or {}
        kyc = person.get("kyc_verified")

        if decision is None:
            # skip invalid lines
            continue

        decision = str(decision).upper()
        total += 1
        shard_name = os.path.basename(shard)
        class_counter[decision] += 1
        by_shard[shard_name][decision] += 1
        decision_counts[decision] += 1
        if esc is not None:
            escalation_counts[esc] += 1
        decision_escal_pairs[(decision, esc)] += 1

        # Reasons quality
        if decision != "CLEAR":
            non_clear_total += 1
            if not reasons:  # missing reasons
                non_clear_empty_reasons += 1
            else:
                # duplicates?
                if len(set(reasons)) < len(reasons):
                    reasons_dup_cases += 1
        for r in reasons:
            reasons_top[str(r)] += 1
        reasons_len_stats.append(len(reasons or []))

        # Fired rules hygiene
        if fired:
            fr_upper = [str(x).upper() for x in fired]
            if any(x in ("INIT", "FINAL_DECISION") for x in fr_upper):
                fired_rules_noise_cases += 1
            if len(set(fr_upper)) < len(fr_upper):
                fired_rules_dup_cases += 1
            fired_rules_len_stats.append(len(fr_upper))
        else:
            fired_rules_len_stats.append(0)

        # Pattern coverage (search in reasons + fired)
        union_texts = (reasons or []) + (fired or [])
        for pname, needles in PATTERNS.items():
            # Special handling for KYC, which is a fact, not always a reason/rule
            is_hit = False
            if pname == "KYC_UNVERIFIED":
                if kyc is False:
                    is_hit = True
            elif contains_any(union_texts, tuple(n.upper() for n in needles)):
                is_hit = True
            if is_hit:
                pattern_hits[pname] += 1
                pattern_decision[pname][decision] += 1
                shard_patterns[shard_name][pname] += 1

        # Odd-hour specific check
        if contains_any(union_texts, ("ODD_HOUR",)):
            odd_hour_total += 1
            if decision == "CLEAR":
                odd_hour_clear += 1

        # KYC unverified coverage
        if kyc is False:
            kyc_false_total += 1
            kyc_false_decision[decision] += 1

    # --- Reporting ---
    if total == 0:
        print("No valid records found."); sys.exit(1)

    print(f"\n=== DATASET QUALITY REPORT ===")
    print(f"Files scanned: {len(files)}")
    print(f"Total records: {total}\n")

    # Class Balance
    print("1) CLASS BALANCE (overall)")
    for cls in ("CLEAR","REVIEW","SAR","BLOCK"):
        frac = class_counter[cls] / total if total else 0.0
        lo, hi = TARGET_BANDS[cls]
        status = band_status(frac, lo, hi)
        print(f"   - {cls:<6}: {class_counter[cls]:>7}  ({pretty_pct(frac)})  target {int(lo*100)}–{int(hi*100)}%  -> {status}")
    print()

    # Per-shard Class Balance
    print("   Per-shard snapshot (first 12 shown):")
    for i, (shard, cnt) in enumerate(sorted(by_shard.items())):
        if i >= 12: break
        shard_total = sum(cnt.values()) or 1
        parts = []
        for cls in ("CLEAR","REVIEW","SAR","BLOCK"):
            frac = cnt[cls]/shard_total
            lo, hi = TARGET_BANDS[cls]
            status = band_status(frac, lo, hi)
            parts.append(f"{cls}:{pretty_pct(frac)}({status})")
        print(f"   - {shard}: {', '.join(parts)}")
    print()

    # Reasons Quality
    print("2) REASONS QUALITY")
    nc = non_clear_total or 1
    print(f"   Non-CLEAR samples: {non_clear_total}")
    print(f"   Non-CLEAR with EMPTY reasons: {non_clear_empty_reasons} ({pretty_pct(non_clear_empty_reasons / nc)})")
    print(f"   Non-CLEAR with DUPLICATE reasons: {reasons_dup_cases} ({pretty_pct(reasons_dup_cases / nc)})")
    if reasons_len_stats:
        print(f"   Reasons length avg/med/max: {mean(reasons_len_stats):.2f} / ~{sorted(reasons_len_stats)[len(reasons_len_stats)//2]} / {max(reasons_len_stats)}")
    print("   Top 10 reasons (raw counts):")
    for r, c in reasons_top.most_common(10):
        print(f"     - {r}: {c}")
    print()

    # Escalation Consistency
    print("3) ESCALATION CONSISTENCY")
    mismatches = 0
    for (dec, esc), c in decision_escal_pairs.items():
        if esc is None:
            mismatches += c
            continue
        expected = CANON_ESC.get(dec)
        if expected is None or esc != expected:
            mismatches += c
    total_pairs = sum(decision_escal_pairs.values()) or 1
    print(f"   Decision↔Escalation mismatches: {mismatches} ({pretty_pct(mismatches/total_pairs)})")
    print("   Decision×Escalation counts (top 10):")
    for (dec, esc), c in decision_escal_pairs.most_common(10):
        print(f"     - {dec} / {esc}: {c}")
    print()

    # Fired rules hygiene
    print("4) FIRED-RULES HYGIENE")
    print(f"   Records with INIT/FINAL_DECISION noise: {fired_rules_noise_cases} ({pretty_pct(fired_rules_noise_cases/total)})")
    print(f"   Records with duplicate fired_rules: {fired_rules_dup_cases} ({pretty_pct(fired_rules_dup_cases/total)})")
    if fired_rules_len_stats:
        print(f"   Fired_rules length avg/med/max: {mean(fired_rules_len_stats):.2f} / ~{sorted(fired_rules_len_stats)[len(fired_rules_len_stats)//2]} / {max(fired_rules_len_stats)}")
    print()

    # Tough pattern coverage
    print("5) COVERAGE OF TOUGH PATTERNS (hits anywhere in reasons or fired_rules)")
    for pname in PATTERNS.keys():
        hits = pattern_hits[pname]
        print(f"   - {pname:<15}: {hits} ({pretty_pct(hits/total)})  by decision: {dict(pattern_decision[pname])}")
    if odd_hour_total:
        print(f"   ODD_HOUR not always suspicious: CLEAR among ODD_HOUR = {odd_hour_clear}/{odd_hour_total} ({pretty_pct(odd_hour_clear/odd_hour_total)})")
    if kyc_false_total:
        print(f"   KYC_UNVERIFIED distribution (total={kyc_false_total}): {dict(kyc_false_decision)}")
    print()

    # Vision & Remediation
    print("6) VISION & ACTIONABLE REMEDIATION")
    # Class balance guidance
    for cls in ("CLEAR","REVIEW","SAR","BLOCK"):
        frac = class_counter[cls]/total
        lo, hi = TARGET_BANDS[cls]
        if frac < lo:
            print(f"   • {cls}: BELOW target — increase scenario quota for {cls} triggers.")
        elif frac > hi:
            print(f"   • {cls}: ABOVE target — reduce scenario quota or tighten rule triggers relating to {cls}.")
    # Reasons guidance
    if non_clear_empty_reasons > 0:
        print("   • Non-CLEAR with empty reasons: fix generator or Drools to always attach at least one reason.")
    if reasons_dup_cases > 0:
        print("   • Duplicate reasons present: de-duplicate in post-processing; use counts if you need evidence (e.g., {\"LARGE_WIRE\":3}).")
    # Escalation mapping guidance
    if mismatches > 0:
        print("   • Decision↔Escalation mismatches: enforce canonical mapping CLEAR=0, REVIEW=1, SAR=2, BLOCK=3 (post-label pass or in Drools).")
    # Fired rules guidance
    if fired_rules_noise_cases > 0 or fired_rules_dup_cases > 0:
        print("   • Fired_rules noise/duplicates: strip INIT/FINAL_DECISION and de-dup before training; keep raw list only for audit.")
    # Pattern coverage
    uncovered = [p for p in PATTERNS if pattern_hits[p] == 0]
    if uncovered:
        print(f"   • Missing pattern coverage: {', '.join(uncovered)} — add scenario quotas to ensure representation.")
    if odd_hour_total and (odd_hour_clear/odd_hour_total) < 0.10:
        print("   • ODD_HOUR always flagged: add some ODD_HOUR cases that are CLEAR to avoid shortcut learning.")
    if kyc_false_total == 0:
        print("   • No KYC_UNVERIFIED coverage: add cases with kyc_verified=false across all decision classes.")

    # Optional JSON dump
    if args.dump_json:
        summary = {
            "total_records": total,
            "class_counts": dict(class_counter),
            "class_fractions": {k: class_counter[k]/total for k in class_counter},
            "target_bands": TARGET_BANDS,
            "non_clear_total": non_clear_total,
            "non_clear_empty_reasons": non_clear_empty_reasons,
            "non_clear_duplicate_reasons_cases": reasons_dup_cases,
            "decision_escalation_pairs": {f"{k[0]}|{k[1]}": v for k, v in decision_escal_pairs.items()},
            "mismatch_count": mismatches,
            "fired_rules_noise_cases": fired_rules_noise_cases,
            "fired_rules_duplicate_cases": fired_rules_dup_cases,
            "pattern_hits": dict(pattern_hits),
            "pattern_by_decision": {p: dict(pattern_decision[p]) for p in pattern_decision},
            "odd_hour": {"total": odd_hour_total, "clear": odd_hour_clear},
            "kyc_unverified": {"total": kyc_false_total, "by_decision": dict(kyc_false_decision)},
            "per_shard_class": {sh: dict(cnt) for sh, cnt in by_shard.items()},
            "per_shard_patterns": {sh: dict(cnt) for sh, cnt in shard_patterns.items()},
        }
        with open(args.dump_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote JSON summary -> {args.dump_json}")

if __name__ == "__main__":
    main()
