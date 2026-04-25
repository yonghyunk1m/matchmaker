"""Comprehensive trajectory pilot: Stage 0 (baseline) + Stage 1 (global retempo).

Saves per-beat trajectory + aggregate metrics for inspection.
If successful, scale to full set.

Output:
  results/comprehensive_trajectory/
    config.json
    metrics_aggregate.csv
    trajectories/{piece}_{perf}_{condition}.csv
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_experiment import PIECES, load_gt
from run_cqt_native import (
    _get_xml_tempo, _build_score_times_for_beats,
    transfer_from_score_to_predicted_perf, SAMPLE_RATE, FRAME_RATE,
    CQTChromagramProcessor, StreamCQTChromagramProcessor,
    OnlineTimeWarpingDixon, Matchmaker, _mm_mod,
)


OUT_ROOT = Path("results/comprehensive_trajectory")
OUT_TRAJ = OUT_ROOT / "trajectories"
OUT_TRAJ.mkdir(parents=True, exist_ok=True)


def run_dixon_cqt(piece, performer, use_resynth=False):
    """Run Dixon+CQT, optionally with global retempo. Returns per-beat dict."""
    piece_dir = piece["asap_dir"]
    xml_path = os.path.join(piece_dir, "xml_score.musicxml")
    wav_path = os.path.join(piece_dir, f"{performer}.wav")
    if not os.path.exists(wav_path):
        return None

    initial_tempo = _get_xml_tempo(xml_path)
    tempo_for_eval = initial_tempo

    if use_resynth:
        _mm_mod.DEFAULT_TEMPO = initial_tempo
        mm0 = Matchmaker(score_file=xml_path, performance_file=wav_path,
                        input_type="audio", wait=False)
        na0 = mm0.score_part.note_array()
        fb0 = na0["onset_beat"].min()
        trim0 = max(fb0, 0) * (60.0 / initial_tempo)
        ts0 = int(trim0 * SAMPLE_RATE)
        if ts0 > 0:
            mm0.score_audio = mm0.score_audio[ts0:]
            mm0.reference_features = mm0.processor(mm0.score_audio)
        mm0.score_follower = OnlineTimeWarpingDixon(
            reference_features=mm0.reference_features, queue=mm0.stream.queue)
        for _ in mm0.run(verbose=False):
            pass
        wp0 = np.array(mm0.score_follower.warping_path)
        if wp0[1][-1] > 0:
            ratio = wp0[0][-1] / wp0[1][-1]
            tempo_for_eval = initial_tempo * ratio

    _mm_mod.DEFAULT_TEMPO = tempo_for_eval
    mm = Matchmaker(score_file=xml_path, performance_file=wav_path,
                    input_type="audio", wait=False)

    na = mm.score_part.note_array()
    first_beat = na["onset_beat"].min()
    bt = mm.score_part.time_sigs[0].beat_type if mm.score_part.time_sigs else 4
    trim = max(first_beat, 0) * (60.0 / tempo_for_eval)
    ts = int(trim * SAMPLE_RATE)
    if ts > 0:
        mm.score_audio = mm.score_audio[ts:]

    # CQT chroma
    mm.processor = CQTChromagramProcessor(sample_rate=SAMPLE_RATE)
    mm.reference_features = mm.processor(mm.score_audio)
    import librosa as _lr
    perf_y, _ = _lr.load(wav_path, sr=SAMPLE_RATE)
    mm.stream.processor = StreamCQTChromagramProcessor(perf_y, sample_rate=SAMPLE_RATE)

    mm.score_follower = OnlineTimeWarpingDixon(
        reference_features=mm.reference_features, queue=mm.stream.queue)
    for _ in mm.run(verbose=False):
        pass
    wp = np.array(mm.score_follower.warping_path)

    gt_pt, gt_bt = load_gt(piece_dir, performer)
    sp_bmin = max(na["onset_beat"].min(), 0.0)
    sp_bmax = na["onset_beat"].max()
    lut_b = np.linspace(sp_bmin, sp_bmax, 2000)
    lut_t = _build_score_times_for_beats(mm.score_part, lut_b, tempo_for_eval)
    gt_beats_scaled = gt_bt * (bt / 4)
    score_annots = np.interp(gt_beats_scaled, lut_b, lut_t) - trim
    ref_dur = len(mm.score_audio) / SAMPLE_RATE
    score_annots = np.clip(score_annots, 0, ref_dur)

    pred = transfer_from_score_to_predicted_perf(wp, score_annots, FRAME_RATE)
    err_ms = np.where(np.isnan(pred), np.nan, np.abs(gt_pt - pred) * 1000.0)

    # Build per-beat dict
    return {
        "beat_idx": np.arange(len(gt_pt)),
        "gt_beat": gt_bt,
        "gt_time_s": gt_pt,
        "pred_time_s": pred,
        "err_ms": err_ms,
        "ok": (~np.isnan(pred)).astype(int),
    }


def metrics_from_err(err_ms):
    err = err_ms[~np.isnan(err_ms)]
    if len(err) == 0:
        return {}
    n = len(err_ms)
    within500 = (err_ms <= 500) & (~np.isnan(err_ms))
    K = 5
    locked = np.zeros(n, dtype=bool)
    for i in range(K - 1, n):
        locked[i] = within500[i - K + 1: i + 1].all()
    return {
        "n_beats": int(len(err_ms)),
        "n_valid": int(len(err)),
        "AR@100": float(np.mean(err_ms <= 100) * 100),
        "AR@250": float(np.mean(err_ms <= 250) * 100),
        "AR@500": float(np.mean(err_ms <= 500) * 100),
        "AR@1000": float(np.mean(err_ms <= 1000) * 100),
        "AR@2000": float(np.mean(err_ms <= 2000) * 100),
        "MedAE_ms": float(np.median(err)),
        "MaxAE_ms": float(np.max(err)),
        "MeanAE_ms": float(np.mean(err)),
        "StdAE_ms": float(np.std(err)),
        "LR500_K5": float(np.sum(locked) / n * 100),
    }


# Pilot scope
PILOT_PIECES = [
    "chopin_ballade1",   # silent-gap cascade
    "chopin_ballade_2",  # cross-piece
    "bach_prelude_848",  # easy baseline
]
PERF_PER_PIECE = 2
CONDITIONS = ["C1_baseline", "C2_global_retempo"]


def main():
    config = {
        "pilot_pieces": PILOT_PIECES,
        "perf_per_piece": PERF_PER_PIECE,
        "conditions": CONDITIONS,
        "stages": {
            "C1_baseline": "Dixon+CQT, no plan, no retempo",
            "C2_global_retempo": "Dixon+CQT, Pass-1 ρ → reference re-render at ρ·T_0",
        },
    }
    with open(OUT_ROOT / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    rows = []
    for pid in PILOT_PIECES:
        piece = next(p for p in PIECES if p["id"] == pid)
        perfs = piece["performers"][:PERF_PER_PIECE]
        for perf in perfs:
            for cond in CONDITIONS:
                use_resynth = (cond == "C2_global_retempo")
                tag = f"{pid}_{perf}_{cond}"
                t0 = time.time()
                try:
                    traj = run_dixon_cqt(piece, perf, use_resynth=use_resynth)
                    if traj is None:
                        print(f"  SKIP {tag}: no audio", flush=True)
                        continue
                    # Save trajectory
                    df = pd.DataFrame({
                        "beat_idx": traj["beat_idx"],
                        "gt_beat": traj["gt_beat"],
                        "gt_time_s": traj["gt_time_s"],
                        "pred_time_s": traj["pred_time_s"],
                        "err_ms": traj["err_ms"],
                        "ok": traj["ok"],
                    })
                    df.to_csv(OUT_TRAJ / f"{tag}.csv", index=False)

                    # Aggregate
                    m = metrics_from_err(traj["err_ms"])
                    elapsed = time.time() - t0
                    row = {"piece": pid, "perf": perf, "condition": cond, "elapsed_s": round(elapsed, 1)}
                    row.update(m)
                    rows.append(row)
                    print(f"  ✓ {tag}: AR@500={m['AR@500']:.1f}% MedAE={m['MedAE_ms']:.0f}ms MaxAE={m['MaxAE_ms']:.0f}ms ({elapsed:.0f}s)", flush=True)
                except Exception as e:
                    print(f"  FAIL {tag}: {e}", flush=True)
    df_agg = pd.DataFrame(rows)
    df_agg.to_csv(OUT_ROOT / "metrics_aggregate.csv", index=False)
    print(f"\nSaved {len(df_agg)} runs to {OUT_ROOT}")


if __name__ == "__main__":
    main()
