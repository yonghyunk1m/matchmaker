"""Multi-metric analysis: does the PerformancePlan help across pieces and metrics?

For each (piece, performer) with both baseline + rule_plan per-beat CSVs:
  Compute: AR@100/500/1000, MedAE, MaxAE, MeanAE, LR@500 K=5
  Report deltas (plan - baseline).

Aggregates: per-piece mean delta, overall mean delta, per-metric significance.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SECTION_DIRS = [
    (Path("results/section_ballade23/per_beat"), "chopin_ballade1"),
    (Path("results/section_extend/per_beat"), None),
]


def load_pair(baseline_csv, plan_csv):
    """Returns (err_ms_baseline, err_ms_plan)."""
    df_b = pd.read_csv(baseline_csv)
    df_p = pd.read_csv(plan_csv)
    return df_b["err_ms"].values, df_p["err_ms"].values


def metrics(err_ms):
    """Compute multi-metrics from per-beat error (ms)."""
    err = err_ms[~np.isnan(err_ms)]
    if len(err) == 0:
        return {}
    return {
        "AR@100": float(np.mean(err <= 100) * 100),
        "AR@500": float(np.mean(err <= 500) * 100),
        "AR@1000": float(np.mean(err <= 1000) * 100),
        "MedAE_ms": float(np.median(err)),
        "MaxAE_ms": float(np.max(err)),
        "MeanAE_ms": float(np.mean(err)),
        "LR500_K5": lr_at(err_ms, tau=500, K=5),
    }


def lr_at(err_ms, tau=500.0, K=5):
    """Lock Rate: fraction of beats where b and preceding K-1 beats all <= tau."""
    n = len(err_ms)
    if n < K:
        return 0.0
    within = (err_ms <= tau) & (~np.isnan(err_ms))
    locked = np.zeros(n, dtype=bool)
    for i in range(K - 1, n):
        locked[i] = within[i - K + 1: i + 1].all()
    return float(np.sum(locked) / n * 100)


def find_pairs(section_dir, default_piece=None):
    """Find (piece, performer, baseline_csv, plan_csv) tuples."""
    pairs = []
    files = list(section_dir.glob("*_baseline.csv"))
    for bf in files:
        stem = bf.name.replace("_baseline.csv", "")
        pf = section_dir / f"{stem}_rule_plan.csv"
        if not pf.exists():
            continue
        # If default_piece given, treat full stem as performer (e.g. ballade23 dir)
        if default_piece is not None:
            piece, perf = default_piece, stem
        else:
            parts = stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            piece, perf = parts
        pairs.append((piece, perf, bf, pf))
    return pairs


def main():
    rows = []
    for sec_dir, default_piece in SECTION_DIRS:
        if not sec_dir.exists():
            continue
        for piece, perf, bf, pf in find_pairs(sec_dir, default_piece):
            try:
                err_b, err_p = load_pair(bf, pf)
            except Exception as e:
                print(f"FAIL {piece}/{perf}: {e}")
                continue
            mb = metrics(err_b)
            mp = metrics(err_p)
            row = {"piece": piece, "perf": perf}
            for k in mb:
                row[f"{k}_base"] = mb[k]
                row[f"{k}_plan"] = mp[k]
                row[f"{k}_delta"] = mp[k] - mb[k]
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        print("No data found.")
        return
    df.to_csv("results/plan_broad_metrics.csv", index=False)
    print(f"Saved {len(df)} pairs to results/plan_broad_metrics.csv\n")

    # Per-piece summary
    print("=== Per-piece mean deltas ===")
    metric_keys = ["AR@100", "AR@500", "AR@1000", "MedAE_ms", "MaxAE_ms", "MeanAE_ms", "LR500_K5"]
    for piece in df["piece"].unique():
        sub = df[df["piece"] == piece]
        n = len(sub)
        print(f"\n  {piece} (n={n}):")
        for k in metric_keys:
            d_col = f"{k}_delta"
            if d_col in sub:
                mean_d = sub[d_col].mean()
                base_mean = sub[f"{k}_base"].mean()
                plan_mean = sub[f"{k}_plan"].mean()
                arrow = "↑" if mean_d > 0 else ("↓" if mean_d < 0 else "=")
                # For error metrics, lower is better
                better = "better" if (k.startswith("AR") or k == "LR500_K5") and mean_d > 0 else (
                         "better" if (not k.startswith("AR")) and k != "LR500_K5" and mean_d < 0 else "")
                print(f"    {k:12s}: base={base_mean:7.1f}, plan={plan_mean:7.1f}, Δ={mean_d:+7.1f} {arrow} {better}")

    # Overall
    print("\n=== Overall (all pieces, n={}) ===".format(len(df)))
    for k in metric_keys:
        d_col = f"{k}_delta"
        if d_col in df:
            mean_d = df[d_col].mean()
            std_d = df[d_col].std()
            base_mean = df[f"{k}_base"].mean()
            plan_mean = df[f"{k}_plan"].mean()
            print(f"  {k:12s}: base={base_mean:7.1f}, plan={plan_mean:7.1f}, Δ={mean_d:+7.1f} ± {std_d:.1f}")


if __name__ == "__main__":
    main()
