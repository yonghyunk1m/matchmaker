"""Best-of-3 selection analysis: per (piece, perf), pick max(AR@500) across
{Pass1 rule, Pass2 gated, Dixon+CQT}. Quantifies the upper bound of
"always ≥ baseline" achievable via rehearsal-time GT comparison.

This is the rehearsal-driven contribution candidate for the paper. Operates
purely on already-collected trajectory CSVs.

Output:
  results/best_of_3/per_perf.csv   - per (piece, perf) selection result
  results/best_of_3/summary.txt     - aggregate stats
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT_ROOT = ROOT / "results" / "best_of_3"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def all_metrics(csv_path):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    err = df["err_ms"].abs()
    if not np.isfinite(err).any():
        return None
    K = 5
    ok = (err <= 500).values
    lr = sum(1 for i in range(K-1, len(ok)) if all(ok[i-K+1:i+1])) / max(len(ok)-K+1, 1) * 100
    return {
        "AR@500": float((err <= 500).mean() * 100),
        "MedAE_s": float(err.median()) / 1000,
        "MaxAE_s": float(err.max()) / 1000,
        "MeanAE_s": float(err.mean()) / 1000,
        "LR_K5": lr,
        "Robust": int(float(err.median()) / 1000 < 2.0),
    }


def main():
    rows = []
    # All Op.~23 + Op.~38 + Bach + ... where both Pass1 and Pass2 trajectories exist
    p1_dir = ROOT / "results/confidence_gated_v2/trajectories"
    p2_dir = ROOT / "results/confidence_gated_v2/trajectories"
    dx_dir = ROOT / "results/hybrid_full/trajectories"

    if not p1_dir.exists():
        print("No confidence-gated v2 results.")
        return

    for p1_csv in sorted(p1_dir.glob("*_pass1_v2.csv")):
        base = p1_csv.name.replace("_pass1_v2.csv", "")
        p2_csv = p2_dir / f"{base}_pass2_v2.csv"
        dx_csv = dx_dir / f"{base}_dixon_cqt.csv"
        if not (p2_csv.exists() and dx_csv.exists()):
            continue
        # Parse piece + perf from base
        tokens = base.split("_")
        perf = tokens[-1]
        piece = "_".join(tokens[:-1])

        m_p1 = all_metrics(p1_csv)
        m_p2 = all_metrics(p2_csv)
        m_dx = all_metrics(dx_csv)
        if not all([m_p1, m_p2, m_dx]):
            continue
        candidates = {"Pass1": m_p1["AR@500"], "Pass2": m_p2["AR@500"], "Dixon+CQT": m_dx["AR@500"]}
        best_k = max(candidates, key=candidates.get)
        best_v = candidates[best_k]
        best_metrics = {"Pass1": m_p1, "Pass2": m_p2, "Dixon+CQT": m_dx}[best_k]
        delta_vs_dx = best_v - m_dx["AR@500"]

        rows.append({
            "piece": piece, "perf": perf,
            "AR_Pass1": round(m_p1["AR@500"], 1), "AR_Pass2": round(m_p2["AR@500"], 1),
            "AR_Dixon": round(m_dx["AR@500"], 1), "best_choice": best_k,
            "best_AR": round(best_v, 1), "delta_vs_dx": round(delta_vs_dx, 2),
            "best_MedAE_s": round(best_metrics["MedAE_s"], 2),
            "best_MaxAE_s": round(best_metrics["MaxAE_s"], 2),
            "best_LR_K5": round(best_metrics["LR_K5"], 1),
            "best_Robust": best_metrics["Robust"],
            "dx_MedAE_s": round(m_dx["MedAE_s"], 2),
            "dx_MaxAE_s": round(m_dx["MaxAE_s"], 2),
            "dx_LR_K5": round(m_dx["LR_K5"], 1),
            "dx_Robust": m_dx["Robust"],
        })

    if not rows:
        print("No complete (pass1, pass2, dixon) triples found.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "per_perf.csv", index=False)

    print(f"Best-of-3 selection across {len(df)} (piece, perf) pairs:\n")
    print(df.to_string(index=False))

    delta = df["delta_vs_dx"]
    n_imp = int((delta > 0.5).sum())
    n_neu = int((delta.abs() <= 0.5).sum())
    n_reg = int((delta < -0.5).sum())
    summary = [
        f"\n=== Aggregate (vs Dixon+CQT baseline) ===",
        f"n = {len(df)}",
        f"improve  (Δ > +0.5): {n_imp}",
        f"neutral (|Δ| ≤ 0.5): {n_neu}",
        f"REGRESS  (Δ < -0.5): {n_reg}    ← \"always ≥ baseline\" claim requires 0",
        f"mean Δ: {delta.mean():+.2f}pp",
        f"min Δ:  {delta.min():+.2f}pp",
        f"max Δ:  {delta.max():+.2f}pp",
        f"",
        f"Best-choice distribution:",
    ]
    for k, v in df["best_choice"].value_counts().items():
        summary.append(f"  {k}: {v}")
    summary.append(f"")
    summary.append(f"Multi-metric improvement (best-of-3 vs Dixon+CQT) summary:")
    for col in ["AR@500", "MedAE_s", "MaxAE_s", "LR_K5"]:
        if col == "AR@500":
            best_col = "best_AR"; dx_col = "AR_Dixon"
        else:
            best_col = f"best_{col}"; dx_col = f"dx_{col}"
        if best_col not in df.columns or dx_col not in df.columns:
            continue
        delta_col = df[best_col] - df[dx_col]
        better = "lower" if col.startswith("Med") or col.startswith("Max") else "higher"
        signed = -delta_col if better == "lower" else delta_col
        n_b = int((signed > 0).sum())
        summary.append(f"  {col}: better in {n_b}/{len(df)} pairs (mean Δ {delta_col.mean():+.3f})")

    print("\n".join(summary))
    (OUT_ROOT / "summary.txt").write_text("\n".join(summary))
    print(f"\nWrote {OUT_ROOT}/per_perf.csv and summary.txt")


if __name__ == "__main__":
    main()
