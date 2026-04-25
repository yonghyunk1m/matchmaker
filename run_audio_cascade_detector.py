"""Audio-only cascade region detection (no score).

Hypothesis: cascade regions (sustained chord + pedal resonance) leave a
distinctive audio fingerprint — high spectral flatness, low onset density,
high chroma stability — that can be detected from the performance audio
alone, without the score's Italian markings.

Method (rule-based proxy, no ML training):
  1. Compute per-beat features from audio:
     - chroma stability (1 - mean rolling cosine-distance over 5s window)
     - onset density (per-beat onset count from librosa)
     - spectral flatness (geometric mean / arithmetic mean)
  2. Combine to single cascade-likelihood score per beat.
  3. Threshold → predicted cascade beats.
  4. Compare against score-marking ground truth (build_directional_plan
     slow_beats).

Output: results/audio_cascade/agreement.csv
  per-piece (precision, recall, F1) of audio detector vs score ground truth.

This is a proof-of-concept that audio features carry the cascade signal,
motivating future learned routers.
"""
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import partitura as pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_experiment import PIECES, load_gt
from run_directional_plan_pilot import build_directional_plan

OUT_ROOT = Path("results/audio_cascade")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050
HOP_LEN = 512
WINDOW_S = 5.0  # rolling window for chroma stability


def compute_audio_features_per_beat(wav_path, gt_pt):
    """gt_pt: array of beat-time-stamps (seconds). Returns per-beat features."""
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LEN)
    spec_flat = librosa.feature.spectral_flatness(y=y, hop_length=HOP_LEN)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LEN)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=HOP_LEN)

    win_frames = int(WINDOW_S * sr / HOP_LEN)
    chroma_stability = np.zeros(chroma.shape[1])
    for i in range(chroma.shape[1]):
        a = max(0, i - win_frames // 2)
        b = min(chroma.shape[1], i + win_frames // 2)
        if b - a < 3:
            chroma_stability[i] = 0
            continue
        seg = chroma[:, a:b]
        # mean cosine distance to centroid: low = stable
        c = seg.mean(axis=1, keepdims=True)
        c_norm = c / (np.linalg.norm(c) + 1e-8)
        seg_norm = seg / (np.linalg.norm(seg, axis=0, keepdims=True) + 1e-8)
        cosines = (seg_norm * c_norm).sum(axis=0)
        chroma_stability[i] = float(np.mean(cosines))

    n_beats = len(gt_pt)
    feats = np.zeros((n_beats, 3))
    for bi, t in enumerate(gt_pt):
        if np.isnan(t):
            continue
        # frame nearest to t
        i = int(np.searchsorted(times, t))
        i = np.clip(i, 0, len(times) - 1)
        a = max(0, i - win_frames // 2)
        b = min(len(times), i + win_frames // 2)
        feats[bi, 0] = float(np.mean(chroma_stability[a:b]))
        feats[bi, 1] = float(np.mean(spec_flat[a:b]))
        feats[bi, 2] = float(np.mean(onset_env[a:b]))
    return feats  # cols: chroma_stab, spec_flat, onset_strength


def cascade_likelihood(feats):
    """Combine features → single cascade likelihood (high = cascade)."""
    chroma_stab = feats[:, 0]   # high in cascade
    spec_flat = feats[:, 1]     # high in sustained tones
    onset = feats[:, 2]         # low in cascade

    z_cs = (chroma_stab - 0.5) / 0.3
    z_sf = (spec_flat - 0.05) / 0.05
    z_on = -(onset - onset.mean()) / (onset.std() + 1e-8)
    return z_cs + z_sf + z_on


def main():
    rows = []
    SUBSET = ["chopin_ballade1", "chopin_ballade_2", "bach_prelude_848",
              "schubert_op31", "liszt_sonata"]
    for piece in PIECES:
        if piece["id"] not in SUBSET:
            continue
        pid = piece["id"]
        xml_path = os.path.join(piece["asap_dir"], "xml_score.musicxml")
        score = pt.load_musicxml(xml_path).parts[0]
        max_beat = int(np.ceil(score.note_array()["onset_beat"].max())) + 10

        # Score ground-truth cascade mask (binary: slow-marked beats)
        gamma_dir, _, _, _ = build_directional_plan(xml_path, max_beat, slow_gamma=0.4)
        gt_cascade_per_beat = (gamma_dir < 1.0)
        n_slow = int(gt_cascade_per_beat.sum())
        print(f"\n{pid}: score-marked slow beats = {n_slow}/{max_beat}", flush=True)

        for perf in piece["performers"][:3]:  # subset for speed
            wav_path = os.path.join(piece["asap_dir"], f"{perf}.wav")
            if not os.path.exists(wav_path):
                continue
            try:
                gt_pt, gt_bt = load_gt(piece["asap_dir"], perf)
                feats = compute_audio_features_per_beat(wav_path, gt_pt)
                like = cascade_likelihood(feats)

                # Map per-perf-beat (gt_bt) to per-score-beat for comparison
                pred_per_beat = np.zeros(max_beat, dtype=bool)
                THRESHOLD = 1.0
                for k, b in enumerate(gt_bt):
                    bi = int(b)
                    if 0 <= bi < max_beat and like[k] > THRESHOLD:
                        pred_per_beat[bi] = True

                tp = int((pred_per_beat & gt_cascade_per_beat).sum())
                fp = int((pred_per_beat & ~gt_cascade_per_beat).sum())
                fn = int((~pred_per_beat & gt_cascade_per_beat).sum())
                tn = int((~pred_per_beat & ~gt_cascade_per_beat).sum())
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                row = {"piece": pid, "perf": perf, "n_slow_score": n_slow,
                       "n_pred_audio": int(pred_per_beat.sum()),
                       "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                       "precision": round(precision, 3), "recall": round(recall, 3),
                       "f1": round(f1, 3)}
                rows.append(row)
                print(f"  {perf:18s}  P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}  "
                      f"(score_slow={n_slow}, audio_pred={pred_per_beat.sum()})", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL {perf}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "agreement.csv", index=False)
    print(f"\nSaved {len(df)} rows to {OUT_ROOT}/agreement.csv")
    if len(df):
        # Per-piece summary
        print("\nPer-piece F1:")
        for pid, g in df.groupby("piece"):
            print(f"  {pid:25s}  mean F1 = {g['f1'].mean():.2f}  (n={len(g)})")


if __name__ == "__main__":
    main()
