"""Multimodal algorithm picker: audio features + score features.

Adds score-derived features to the audio-only classifier:
  - n_slow_markings_total (count of Lento, Adagio, Meno mosso, etc.)
  - n_all_markings (all tempo directions)
  - slow_beat_ratio (slow_beats / total_beats per quantitative lexicon)
  - max_slow_run (longest contiguous slow region in beats)
  - n_tempo_changes (rit, accel, ten count)
  - piece_duration_score (total beats / 120 BPM proxy)
  - n_distinct_marking_types

These are computed from MusicXML alone (no audio required for SCORE side).
At deployment: extract score features once per piece, audio features
per concert (real-time). Combined feature vector → classifier → algo pick.
"""
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import partitura as pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_experiment import PIECES
from run_directional_plan_pilot import is_slow_marking
from run_scorpion_rule_intent import extract_rule_gamma
from train_algo_picker import extract_features as extract_audio_features

OUT_ROOT = Path("results/algo_picker_v2")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def extract_score_features(xml_path):
    score = pt.load_musicxml(xml_path).parts[0]
    na = score.note_array()
    max_beat = int(np.ceil(na["onset_beat"].max())) + 10
    gamma_rule, markings = extract_rule_gamma(xml_path, max_beat)
    slow_beats = int(np.sum(gamma_rule < 0.85))

    slow_marks = [(b, txt, g) for (b, kind, txt, g) in markings if is_slow_marking(txt)]
    all_marks = [(b, txt, g) for (b, kind, txt, g) in markings]
    n_slow = len(slow_marks)
    n_total = len(all_marks)
    n_changes = sum(1 for (b, txt, g) in all_marks if any(t in txt.lower() for t in ["rit", "accel", "ten", "rall"]))
    distinct_types = len({txt.lower() for (b, txt, g) in all_marks})

    # Longest contiguous run with γ_rule < 0.85
    is_slow = (gamma_rule < 0.85)
    if is_slow.any():
        # Find runs
        runs = []
        i = 0
        while i < len(is_slow):
            if is_slow[i]:
                j = i
                while j < len(is_slow) and is_slow[j]:
                    j += 1
                runs.append(j - i)
                i = j
            else:
                i += 1
        max_run = max(runs) if runs else 0
    else:
        max_run = 0

    return {
        "score_n_slow_markings": n_slow,
        "score_n_all_markings": n_total,
        "score_slow_beat_ratio": slow_beats / max(max_beat, 1),
        "score_max_slow_run": max_run,
        "score_n_tempo_changes": n_changes,
        "score_total_beats": max_beat,
        "score_n_distinct_marking_types": distinct_types,
    }


def build_multimodal_dataset():
    HYBRID_DIR = Path("results/hybrid_full/trajectories")
    rows = []
    seen = set()
    for hyb_csv in sorted(HYBRID_DIR.glob("*_hybrid.csv")):
        base = hyb_csv.name.replace("_hybrid.csv", "")
        dix_csv = HYBRID_DIR / f"{base}_dixon_cqt.csv"
        if not dix_csv.exists():
            continue
        piece_id, perf = None, None
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
        piece_obj = next(p for p in PIECES if p["id"] == piece_id)
        wav = os.path.join(piece_obj["asap_dir"], f"{perf}.wav")
        xml = os.path.join(piece_obj["asap_dir"], "xml_score.musicxml")
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
            af = extract_audio_features(wav)
            sf = extract_score_features(xml)
            if af is None:
                continue
            row = {"piece": piece_id, "perf": perf, "winner": winner, "delta": delta,
                   "ar_hyb": ar_h, "ar_dix": ar_d}
            row.update(af)
            row.update(sf)
            rows.append(row)
            print(f"  ✓ {base:50s}  winner={winner:6s}  Δ={delta:+6.2f}pp  ({time.time()-t0:.1f}s)")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAIL {base}: {e}")
    return pd.DataFrame(rows)


def lopo_cv(df, feature_cols, label="MM"):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    pieces = sorted(df["piece"].unique())
    if len(pieces) < 2:
        print("Need ≥ 2 pieces")
        return None
    X = df[feature_cols].values
    y = (df["winner"] == "hybrid").astype(int).values
    scaler = StandardScaler()
    correct, preds, truths = [], [], []
    for held in pieces:
        mask = df["piece"] == held
        if not mask.any():
            continue
        Xtr = scaler.fit_transform(X[~mask])
        Xte = scaler.transform(X[mask])
        ytr = y[~mask]
        yte = y[mask]
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1, max_depth=8)
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)
        correct.extend((ypred == yte).tolist())
        preds.extend(ypred.tolist())
        truths.extend(yte.tolist())
    acc = float(np.mean(correct))
    truths = np.array(truths); preds = np.array(preds)
    print(f"\n[{label}] LOPO CV accuracy: {acc*100:.1f}% (n={len(correct)})")

    df_eval = df.copy().reset_index(drop=True)
    df_eval["pred_winner"] = ["hybrid" if p == 1 else "dixon" for p in preds]
    df_eval["pred_AR"] = np.where(df_eval["pred_winner"] == "hybrid",
                                   df_eval["ar_hyb"], df_eval["ar_dix"])
    df_eval["delta_vs_dix_baseline"] = df_eval["pred_AR"] - df_eval["ar_dix"]
    n_reg = int((df_eval["delta_vs_dix_baseline"] < -0.5).sum())
    n_neu = int((df_eval["delta_vs_dix_baseline"].abs() <= 0.5).sum())
    n_imp = int((df_eval["delta_vs_dix_baseline"] > 0.5).sum())
    print(f"  picked_AR ≥ Dixon+CQT: {n_neu+n_imp}/{len(df_eval)} ({(n_neu+n_imp)/len(df_eval)*100:.0f}%)")
    print(f"  REGRESSIONS: {n_reg}/{len(df_eval)}")
    if n_reg > 0:
        print(f"  worst regress mean Δ: "
              f"{df_eval[df_eval['delta_vs_dix_baseline']<-0.5]['delta_vs_dix_baseline'].mean():+.2f}pp")
    print(f"  Mean Δ overall: {df_eval['delta_vs_dix_baseline'].mean():+.2f}pp")
    return acc, df_eval


def main():
    print("=== Building multimodal dataset (audio + score features) ===\n")
    df = build_multimodal_dataset()
    if df.empty:
        print("No data.")
        return
    df.to_csv(OUT_ROOT / "feature_table.csv", index=False)
    print(f"\nSaved {len(df)} rows; pieces={df['piece'].unique().tolist()}")
    print(f"Winners: hybrid={int((df['winner']=='hybrid').sum())}, dixon={int((df['winner']=='dixon').sum())}")

    audio_cols = [c for c in df.columns if c not in {"piece", "perf", "winner", "delta", "ar_hyb", "ar_dix"}
                  and not c.startswith("score_")]
    score_cols = [c for c in df.columns if c.startswith("score_")]
    all_cols = audio_cols + score_cols

    print(f"\nFeature counts: audio={len(audio_cols)}, score={len(score_cols)}, total={len(all_cols)}")

    print(f"\n=== Audio-only ===")
    lopo_cv(df, audio_cols, "AUDIO")
    print(f"\n=== Score-only ===")
    lopo_cv(df, score_cols, "SCORE")
    print(f"\n=== Multimodal (audio + score) ===")
    lopo_cv(df, all_cols, "MM")


if __name__ == "__main__":
    main()
