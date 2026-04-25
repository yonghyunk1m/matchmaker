"""Train an audio-feature-based classifier to pick the better algorithm
(hybrid Arzt+plan vs Dixon+CQT) per (piece, performer).

Goal: replace ground-truth-comparison-based fallback with audio-only
prediction → fully deployable "always-improve" without rehearsal MIDI.

Pipeline:
  1. For each (piece, perf) where we have both hybrid_full results:
     - Extract audio features from performance wav (chroma stats, spectral
       flatness, onset density, tempo estimate, etc.)
     - Label = winner ("hybrid" or "dixon")
  2. Train classifier (RF + MLP) with leave-one-piece-out CV
  3. Compare to:
     - Random baseline (50%)
     - Always-pick-Dixon (majority class)
     - Hand-crafted proxy P4 (66.7% from earlier audio_proxy run)
  4. If LOPO accuracy > 80%, "deployable" claim is supportable.

Output:
  results/algo_picker/feature_table.csv  — per-pair features + labels
  results/algo_picker/lopo_cv.txt        — LOPO accuracy report
  results/algo_picker/confusion.txt      — confusion matrix
"""
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_experiment import PIECES

OUT_ROOT = Path("results/algo_picker")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050
HOP_LEN = 512


def extract_features(wav_path):
    """Extract a fixed-size feature vector from a performance wav file.

    Features (15 dims):
      - chroma_stab_mean, chroma_stab_std (window 5s rolling cosine to centroid)
      - spec_flat_mean, spec_flat_std
      - spec_centroid_mean, spec_centroid_std (Hz)
      - onset_density (events/sec)
      - rms_mean, rms_std
      - tempo_estimate (BPM, librosa)
      - duration_sec
      - silent_frac (fraction of frames with rms < threshold)
      - chroma_entropy_mean (Shannon entropy of chroma vector, mean over time)
      - low_chroma_ratio (energy in pitch classes 0-3 / total)
      - high_chroma_ratio (energy in pitch classes 4-11 / total)
    """
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    duration = len(y) / sr
    if duration < 1.0:
        return None

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LEN)
    spec_flat = librosa.feature.spectral_flatness(y=y, hop_length=HOP_LEN)[0]
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LEN)[0]
    rms = librosa.feature.rms(y=y, hop_length=HOP_LEN)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LEN)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP_LEN)
    onset_density = len(onset_frames) / duration

    # Chroma stability over 5s rolling window
    win = int(5 * sr / HOP_LEN)
    cs = np.zeros(chroma.shape[1])
    for i in range(chroma.shape[1]):
        a = max(0, i - win // 2)
        b = min(chroma.shape[1], i + win // 2)
        if b - a < 3:
            cs[i] = 0
            continue
        seg = chroma[:, a:b]
        c = seg.mean(axis=1, keepdims=True)
        cn = c / (np.linalg.norm(c) + 1e-8)
        sn = seg / (np.linalg.norm(seg, axis=0, keepdims=True) + 1e-8)
        cs[i] = float(np.mean((sn * cn).sum(axis=0)))

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LEN)
        tempo_est = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    except Exception:
        tempo_est = 0.0

    rms_thresh = float(np.percentile(rms, 5)) * 2
    silent = float((rms < rms_thresh).mean())

    chroma_norm = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-8)
    chroma_entropy = -(chroma_norm * np.log(chroma_norm + 1e-12)).sum(axis=0)
    low_e = chroma[:4, :].sum() / (chroma.sum() + 1e-8)
    high_e = chroma[4:, :].sum() / (chroma.sum() + 1e-8)

    return {
        "chroma_stab_mean": float(cs.mean()),
        "chroma_stab_std": float(cs.std()),
        "spec_flat_mean": float(spec_flat.mean()),
        "spec_flat_std": float(spec_flat.std()),
        "spec_centroid_mean": float(spec_cent.mean()),
        "spec_centroid_std": float(spec_cent.std()),
        "onset_density": float(onset_density),
        "rms_mean": float(rms.mean()),
        "rms_std": float(rms.std()),
        "tempo_estimate": tempo_est,
        "duration_sec": float(duration),
        "silent_frac": silent,
        "chroma_entropy_mean": float(chroma_entropy.mean()),
        "low_chroma_ratio": float(low_e),
        "high_chroma_ratio": float(high_e),
    }


def build_dataset():
    """For every (piece, perf) with hybrid + baseline trajectory CSVs,
    extract features + label."""
    HYBRID_DIR = Path("results/hybrid_full/trajectories")
    rows = []
    seen = set()
    for hyb_csv in sorted(HYBRID_DIR.glob("*_hybrid.csv")):
        base = hyb_csv.name.replace("_hybrid.csv", "")
        dix_csv = HYBRID_DIR / f"{base}_dixon_cqt.csv"
        if not dix_csv.exists():
            continue
        # Locate piece + perf
        piece_id = None
        perf = None
        for p in PIECES:
            for pf in p["performers"]:
                if base == f"{p['id']}_{pf}":
                    piece_id, perf = p["id"], pf
                    break
            if piece_id:
                break
        if piece_id is None:
            continue
        if (piece_id, perf) in seen:
            continue
        seen.add((piece_id, perf))
        wav = os.path.join(next(p for p in PIECES if p["id"] == piece_id)["asap_dir"], f"{perf}.wav")
        if not os.path.exists(wav):
            continue
        df_h = pd.read_csv(hyb_csv)
        df_d = pd.read_csv(dix_csv)
        ar_h = float((df_h["err_ms"].abs() <= 500).mean() * 100)
        ar_d = float((df_d["err_ms"].abs() <= 500).mean() * 100)
        winner = "hybrid" if ar_h > ar_d else "dixon"
        delta = ar_h - ar_d
        try:
            t0 = time.time()
            feats = extract_features(wav)
            if feats is None:
                continue
            row = {"piece": piece_id, "perf": perf, "winner": winner, "delta": delta,
                   "ar_hyb": ar_h, "ar_dix": ar_d}
            row.update(feats)
            rows.append(row)
            print(f"  ✓ {base:50s}  winner={winner:6s}  Δ={delta:+6.2f}pp  ({time.time()-t0:.1f}s)")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAIL {base}: {e}")
    return pd.DataFrame(rows)


def lopo_cv(df, feature_cols):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    pieces = sorted(df["piece"].unique())
    if len(pieces) < 2:
        print("Need ≥ 2 pieces for LOPO CV")
        return None
    X = df[feature_cols].values
    y = (df["winner"] == "hybrid").astype(int).values
    scaler = StandardScaler()
    correct = []
    preds = []
    truths = []
    for held in pieces:
        mask = df["piece"] == held
        if not mask.any():
            continue
        Xtr = scaler.fit_transform(X[~mask])
        Xte = scaler.transform(X[mask])
        ytr = y[~mask]
        yte = y[mask]
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)
        correct.extend((ypred == yte).tolist())
        preds.extend(ypred.tolist())
        truths.extend(yte.tolist())
        print(f"  fold (held={held}): n_test={mask.sum()}, "
              f"acc={(ypred==yte).mean()*100:.1f}%")
    acc = float(np.mean(correct))
    print(f"\nLOPO CV accuracy: {acc*100:.1f}% (n={len(correct)})")
    print(f"Always-pick-Dixon (majority): {(1-np.array(truths)).mean()*100:.1f}%")
    print(f"Always-pick-Hybrid:          {np.array(truths).mean()*100:.1f}%")
    # Confusion
    truths = np.array(truths); preds = np.array(preds)
    print(f"\nConfusion (rows=true, cols=pred):")
    print(f"            Pred dixon  Pred hybrid")
    print(f"True dixon  {((truths==0)&(preds==0)).sum():>10}  {((truths==0)&(preds==1)).sum():>11}")
    print(f"True hybrid {((truths==1)&(preds==0)).sum():>10}  {((truths==1)&(preds==1)).sum():>11}")

    # Show "what-if-deploy" outcome: if we'd used the predicted algo,
    # how many regressions would remain?
    print(f"\nDeployment simulation:")
    df_eval = df.copy()
    df_eval["pred_winner"] = ["hybrid" if p == 1 else "dixon" for p in preds]
    df_eval["pred_AR"] = np.where(df_eval["pred_winner"] == "hybrid",
                                   df_eval["ar_hyb"], df_eval["ar_dix"])
    df_eval["delta_vs_dix_baseline"] = df_eval["pred_AR"] - df_eval["ar_dix"]
    n_reg = int((df_eval["delta_vs_dix_baseline"] < -0.5).sum())
    n_neu = int((df_eval["delta_vs_dix_baseline"].abs() <= 0.5).sum())
    n_imp = int((df_eval["delta_vs_dix_baseline"] > 0.5).sum())
    print(f"  picked_AR ≥ Dixon+CQT: {n_neu+n_imp}/{len(df_eval)} pairs ({(n_neu+n_imp)/len(df_eval)*100:.0f}%)")
    print(f"  REGRESSIONS:  {n_reg}/{len(df_eval)} pairs (mean Δ when regressed: "
          f"{df_eval[df_eval['delta_vs_dix_baseline']<-0.5]['delta_vs_dix_baseline'].mean():+.2f}pp)")
    print(f"  Mean Δ overall: {df_eval['delta_vs_dix_baseline'].mean():+.2f}pp")
    return acc, df_eval


def main():
    print("=== Building dataset (extracting audio features per perf) ===\n")
    df = build_dataset()
    if df.empty:
        print("No paired data found.")
        return
    df.to_csv(OUT_ROOT / "feature_table.csv", index=False)
    print(f"\nSaved {len(df)} rows to {OUT_ROOT}/feature_table.csv")
    print(f"Pieces: {df['piece'].unique().tolist()}")
    print(f"Winners: hybrid={int((df['winner']=='hybrid').sum())}, dixon={int((df['winner']=='dixon').sum())}")

    feature_cols = [c for c in df.columns if c not in
                    {"piece", "perf", "winner", "delta", "ar_hyb", "ar_dix"}]
    print(f"\n=== LOPO CV (Random Forest, n_estimators=100) ===\n")
    res = lopo_cv(df, feature_cols)


if __name__ == "__main__":
    main()
