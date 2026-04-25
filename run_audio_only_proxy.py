"""Validate audio-only confidence proxies for algorithm selection.

Hypothesis: a tracker's internal state (without ground truth) carries
enough signal to pick the better of {Arzt+plan, Dixon+CQT} for each
performer. If true, "always-improve" via algorithm selection becomes
fully deployable (no rehearsal MIDI needed).

Proxies tested (computed from per-beat trajectory CSV):
  P1: prediction-step variance — std(diff(pred_time_s))  (lower = stable)
  P2: median forward step       — median(diff(pred_time_s))  (close to 1/perf-tempo = good)
  P3: outlier rate              — fraction of |diff(pred_time_s)| > 3s  (high = drifted)
  P4: monotonicity              — fraction of diff(pred_time_s) > 0  (low = backtracking)

Validation: do these proxies correctly predict which algorithm has higher
AR@500 for each (piece, perf)?

Output: results/audio_proxy/agreement.csv — per-pair (proxy values, AR@500, winner).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_ROOT = Path("results/audio_proxy")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def proxies_from_traj(csv_path):
    df = pd.read_csv(csv_path)
    if len(df) < 5:
        return None
    pred = df["pred_time_s"].values
    pred = pred[~np.isnan(pred)]
    if len(pred) < 5:
        return None
    diffs = np.diff(pred)
    finite = np.isfinite(diffs)
    if finite.sum() < 5:
        return None
    diffs = diffs[finite]
    return {
        "P1_step_std": float(np.std(diffs)),
        "P2_step_median": float(np.median(diffs)),
        "P3_outlier_rate": float(np.mean(np.abs(diffs) > 3.0)),
        "P4_monotone_frac": float(np.mean(diffs > 0)),
        "AR@500": float((df["err_ms"].abs() <= 500).mean() * 100),
    }


def main():
    pairs = []
    HYBRID_DIR = Path("results/hybrid_full/trajectories")
    for hyb_csv in sorted(HYBRID_DIR.glob("*_hybrid.csv")):
        base = hyb_csv.name.replace("_hybrid.csv", "")
        dix_csv = HYBRID_DIR / f"{base}_dixon_cqt.csv"
        if not dix_csv.exists():
            continue
        h = proxies_from_traj(hyb_csv)
        d = proxies_from_traj(dix_csv)
        if h is None or d is None:
            continue
        # piece, perf from base
        parts = base.rsplit("_", 1)
        piece, perf = parts[0], parts[1]
        true_winner = "hybrid" if h["AR@500"] > d["AR@500"] else "dixon"
        # Predict winner using each proxy
        p1_winner = "hybrid" if h["P1_step_std"] < d["P1_step_std"] else "dixon"
        p3_winner = "hybrid" if h["P3_outlier_rate"] < d["P3_outlier_rate"] else "dixon"
        p4_winner = "hybrid" if h["P4_monotone_frac"] > d["P4_monotone_frac"] else "dixon"
        pairs.append({
            "piece": piece, "perf": perf,
            "AR_hyb": h["AR@500"], "AR_dix": d["AR@500"],
            "delta_AR": h["AR@500"] - d["AR@500"],
            "true_winner": true_winner,
            # Proxy values
            "P1_std_hyb": h["P1_step_std"], "P1_std_dix": d["P1_step_std"],
            "P3_outl_hyb": h["P3_outlier_rate"], "P3_outl_dix": d["P3_outlier_rate"],
            "P4_mono_hyb": h["P4_monotone_frac"], "P4_mono_dix": d["P4_monotone_frac"],
            # Predictions
            "P1_pred_winner": p1_winner,
            "P3_pred_winner": p3_winner,
            "P4_pred_winner": p4_winner,
            "P1_correct": p1_winner == true_winner,
            "P3_correct": p3_winner == true_winner,
            "P4_correct": p4_winner == true_winner,
        })
    df = pd.DataFrame(pairs)
    df.to_csv(OUT_ROOT / "agreement.csv", index=False)
    if len(df) == 0:
        print("No pairs found.")
        return
    print(f"Pairs evaluated: {len(df)}")
    print(f"\nTrue winner distribution:")
    print(f"  hybrid: {(df['true_winner']=='hybrid').sum()}")
    print(f"  dixon:  {(df['true_winner']=='dixon').sum()}")

    print(f"\nProxy accuracy (correctly identify true winner):")
    for px in ["P1", "P3", "P4"]:
        col = f"{px}_correct"
        acc = df[col].mean()
        print(f"  {px}: {acc*100:.1f}% ({df[col].sum()}/{len(df)})")

    print(f"\nIf proxy correct → 'always ≥ baseline' achievable (just pick the true winner).")
    print(f"Per-pair proxy results:")
    cols = ["piece", "perf", "AR_hyb", "AR_dix", "delta_AR", "true_winner",
            "P1_pred_winner", "P3_pred_winner", "P4_pred_winner",
            "P1_correct", "P3_correct", "P4_correct"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
