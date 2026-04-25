"""Algo picker v3: class-weighted multimodal classifier with cost-sensitive
decision threshold tuning. Optimized for "always-improve" deployment.

Cost analysis:
  - Misclassify "hybrid winner" as dixon → MISSED GAIN (small cost; baseline value)
  - Misclassify "dixon winner" as hybrid → REGRESSION (large cost; below baseline)

→ We want HIGH PRECISION on hybrid predictions (only pick hybrid when very
   confident). This is achieved by:
   (a) class_weight='balanced' (penalize false-hybrid more under imbalance)
   (b) decision threshold > 0.5 for "predict hybrid"
   (c) sweep thresholds and report Pareto curve

Pipeline reuses train_algo_picker_v2.build_multimodal_dataset.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_algo_picker_v2 import build_multimodal_dataset

OUT_ROOT = Path("results/algo_picker_v3")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def lopo_cv_thresholded(df, feature_cols, label, threshold=0.5, class_weight='balanced'):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    pieces = sorted(df["piece"].unique())
    if len(pieces) < 2:
        return None
    X = df[feature_cols].values
    y = (df["winner"] == "hybrid").astype(int).values
    scaler = StandardScaler()
    proba_all = np.zeros(len(df))
    pieces_held = []
    for held in pieces:
        mask = (df["piece"] == held).values
        if not mask.any():
            continue
        Xtr = scaler.fit_transform(X[~mask])
        Xte = scaler.transform(X[mask])
        ytr = y[~mask]
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1,
                                     max_depth=8, class_weight=class_weight)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        # Class 1 = hybrid
        if proba.shape[1] == 1:
            # All training labels were single class → predict that class
            proba_all[mask] = clf.classes_[0]
        else:
            proba_all[mask] = proba[:, 1]
        pieces_held.append(held)
    preds = (proba_all >= threshold).astype(int)
    truths = y
    acc = float(np.mean(preds == truths))
    df_eval = df.copy().reset_index(drop=True)
    df_eval["proba_hybrid"] = proba_all
    df_eval["pred_winner"] = ["hybrid" if p == 1 else "dixon" for p in preds]
    df_eval["pred_AR"] = np.where(df_eval["pred_winner"] == "hybrid",
                                   df_eval["ar_hyb"], df_eval["ar_dix"])
    df_eval["delta_vs_dix_baseline"] = df_eval["pred_AR"] - df_eval["ar_dix"]
    n_reg = int((df_eval["delta_vs_dix_baseline"] < -0.5).sum())
    n_neu = int((df_eval["delta_vs_dix_baseline"].abs() <= 0.5).sum())
    n_imp = int((df_eval["delta_vs_dix_baseline"] > 0.5).sum())
    mean_d = float(df_eval["delta_vs_dix_baseline"].mean())
    # Precision/recall on hybrid class
    n_pred_hyb = int(preds.sum())
    n_true_hyb = int(truths.sum())
    tp = int(((preds == 1) & (truths == 1)).sum())
    fp = int(((preds == 1) & (truths == 0)).sum())
    fn = int(((preds == 0) & (truths == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"[{label}, t={threshold:.2f}]  acc={acc*100:5.1f}%  "
          f"hyb-prec={prec:.2f}  hyb-rec={rec:.2f}  "
          f"REG={n_reg:>3}  NEU={n_neu:>3}  IMP={n_imp:>3}  mean Δ={mean_d:+.2f}pp")
    return {"label": label, "threshold": threshold, "acc": acc,
            "hyb_precision": prec, "hyb_recall": rec,
            "n_regress": n_reg, "n_neutral": n_neu, "n_improve": n_imp,
            "mean_delta": mean_d, "df_eval": df_eval}


def main():
    print("=== Building multimodal dataset ===\n")
    df = build_multimodal_dataset()
    if df.empty:
        print("No data.")
        return
    df.to_csv(OUT_ROOT / "feature_table.csv", index=False)
    print(f"\n{len(df)} rows; pieces={df['piece'].unique().tolist()}")
    print(f"Imbalance: hybrid={int((df['winner']=='hybrid').sum())}, "
          f"dixon={int((df['winner']=='dixon').sum())}")

    audio_cols = [c for c in df.columns if c not in {"piece", "perf", "winner", "delta", "ar_hyb", "ar_dix"}
                  and not c.startswith("score_")]
    score_cols = [c for c in df.columns if c.startswith("score_")]
    all_cols = audio_cols + score_cols

    print(f"\nFeature counts: audio={len(audio_cols)}, score={len(score_cols)}, total={len(all_cols)}")

    print(f"\n=== Threshold sweep (hybrid prediction probability) ===")
    print(f"Cost-sensitive goal: minimize REG (regressions) while maximizing IMP (improvements).\n")

    summaries = []
    for label, cols in [("AUDIO", audio_cols), ("SCORE", score_cols), ("MM", all_cols)]:
        print(f"\n--- Modality: {label} ({len(cols)} features) ---")
        for t in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
            res = lopo_cv_thresholded(df, cols, label, threshold=t)
            if res:
                summaries.append({k: v for k, v in res.items() if k != "df_eval"})

    pd.DataFrame(summaries).to_csv(OUT_ROOT / "threshold_sweep.csv", index=False)
    print(f"\nSaved threshold sweep to {OUT_ROOT}/threshold_sweep.csv")
    print("\n=== Best 'always-improve' candidates (REG=0 with mean Δ > 0) ===")
    best = pd.DataFrame(summaries)
    qualifying = best[(best["n_regress"] == 0) & (best["mean_delta"] > 0)]
    if len(qualifying):
        print(qualifying.to_string(index=False))
    else:
        print("No (label, threshold) achieves REG=0 with positive mean Δ on current data.")
        print("Best by 'regression count then mean delta':")
        print(best.nsmallest(5, "n_regress").nlargest(5, "mean_delta").to_string(index=False))


if __name__ == "__main__":
    main()
