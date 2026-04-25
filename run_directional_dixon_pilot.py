"""Directional Plan via Dixon Path B (variable-tempo reference, slow-only stretch).

For Dixon (which has no step-size knob), inject directional γ via Path B:
  γ_directional(b) = 0.4 if marking ∈ slow_terms else 1.0
  Render reference audio with this γ as tempo_map.

Key: γ=1 default → audio at base_tempo (identical to standard render)
     γ=0.4 slow → audio stretched only in slow regions
  → Non-cascade pieces: γ uniform=1 → bit-identical to baseline → no harm

Conditions:
  C1: Dixon+CQT baseline (no plan)
  C6: Dixon+CQT + directional Path B (γ=0.4 in slow, 1.0 elsewhere)

Pilot: 3 pieces × 2 perfs × 2 conditions = 12 runs.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import partitura as pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_experiment import (
    PIECES, load_gt, SAMPLE_RATE, _get_xml_tempo,
    generate_score_audio_variable,
    _build_score_times_for_beats,
)
from matchmaker import Matchmaker
from matchmaker.dp import OnlineTimeWarpingDixon
from matchmaker.utils.eval import transfer_from_score_to_predicted_perf
from matchmaker.features.audio import FRAME_RATE
from run_cqt_native import CQTChromagramProcessor, StreamCQTChromagramProcessor
import matchmaker.matchmaker as _mm_mod
import librosa

from run_directional_plan_pilot import (
    SLOW_TERMS, is_slow_marking, build_directional_plan, metrics_from_err,
    PILOT,
)

OUT_ROOT = Path("results/directional_plan")
OUT_TRAJ = OUT_ROOT / "trajectories"
OUT_TRAJ.mkdir(parents=True, exist_ok=True)


def gamma_to_tempo_map(gamma, base_tempo):
    """Convert γ array to tempo_map (compress runs of constant γ)."""
    tempo_map = [(0.0, base_tempo * float(gamma[0]))]
    for i in range(1, len(gamma)):
        if gamma[i] != gamma[i - 1]:
            tempo_map.append((float(i), base_tempo * float(gamma[i])))
    return tempo_map


def run_dixon(piece, performer, plan_gamma=None):
    """Dixon+CQT, optionally with directional Path B (variable-tempo reference)."""
    piece_dir = piece["asap_dir"]
    xml_path = os.path.join(piece_dir, "xml_score.musicxml")
    wav_path = os.path.join(piece_dir, f"{performer}.wav")
    if not os.path.exists(wav_path):
        return None

    initial_tempo = _get_xml_tempo(xml_path)
    _mm_mod.DEFAULT_TEMPO = initial_tempo

    score_audio_override = None
    if plan_gamma is not None and np.any(plan_gamma != 1.0):
        score_part = pt.load_musicxml(xml_path).parts[0]
        tempo_map = gamma_to_tempo_map(plan_gamma, initial_tempo)
        score_audio_override = generate_score_audio_variable(
            score_part, tempo_map, SAMPLE_RATE)

    mm = Matchmaker(score_file=xml_path, performance_file=wav_path,
                    input_type="audio", wait=False)

    na = mm.score_part.note_array()
    first_beat = na["onset_beat"].min()
    bt = mm.score_part.time_sigs[0].beat_type if mm.score_part.time_sigs else 4
    trim = max(first_beat, 0) * (60.0 / initial_tempo)
    ts = int(trim * SAMPLE_RATE)
    if score_audio_override is not None:
        mm.score_audio = score_audio_override
    elif ts > 0:
        mm.score_audio = mm.score_audio[ts:]

    # CQT
    mm.processor = CQTChromagramProcessor(sample_rate=SAMPLE_RATE)
    mm.reference_features = mm.processor(mm.score_audio)
    perf_y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
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
    lut_t = _build_score_times_for_beats(mm.score_part, lut_b, initial_tempo)
    gt_beats_scaled = gt_bt * (bt / 4)
    score_annots = np.interp(gt_beats_scaled, lut_b, lut_t) - trim
    ref_dur = len(mm.score_audio) / SAMPLE_RATE
    score_annots = np.clip(score_annots, 0, ref_dur)

    pred = transfer_from_score_to_predicted_perf(wp, score_annots, FRAME_RATE)
    err_ms = np.where(np.isnan(pred), np.nan, np.abs(gt_pt - pred) * 1000.0)

    return {
        "beat_idx": np.arange(len(gt_pt)),
        "gt_beat": gt_bt,
        "gt_time_s": gt_pt,
        "pred_time_s": pred,
        "err_ms": err_ms,
        "ok": (~np.isnan(pred)).astype(int),
    }


def main():
    rows = []
    for pid, n_perf in PILOT:
        try:
            piece = next(p for p in PIECES if p["id"] == pid)
        except StopIteration:
            print(f"  SKIP {pid}: not in PIECES")
            continue
        perfs = piece["performers"][:n_perf]
        xml_path = os.path.join(piece["asap_dir"], "xml_score.musicxml")
        score = pt.load_musicxml(xml_path).parts[0]
        max_beat = int(np.ceil(score.note_array()["onset_beat"].max())) + 10
        gamma, _, _, n_slow = build_directional_plan(xml_path, max_beat, slow_gamma=0.4)
        slow_count = int(np.sum(gamma != 1.0))
        print(f"\n{pid}: max_beat={max_beat}, slow markings={n_slow}, slow beats={slow_count}", flush=True)

        for perf in perfs:
            for cond, plan in [("C1_dixon_baseline", None), ("C6_dixon_directional", gamma)]:
                tag = f"{pid}_{perf}_{cond}"
                t0 = time.time()
                try:
                    traj = run_dixon(piece, perf, plan_gamma=plan)
                    if traj is None:
                        print(f"  SKIP {tag}: no audio", flush=True)
                        continue
                    df = pd.DataFrame({
                        "beat_idx": traj["beat_idx"],
                        "gt_beat": traj["gt_beat"],
                        "gt_time_s": traj["gt_time_s"],
                        "pred_time_s": traj["pred_time_s"],
                        "err_ms": traj["err_ms"],
                        "ok": traj["ok"],
                    })
                    df.to_csv(OUT_TRAJ / f"{tag}.csv", index=False)
                    m = metrics_from_err(traj["err_ms"])
                    elapsed = time.time() - t0
                    row = {"piece": pid, "perf": perf, "condition": cond,
                           "elapsed_s": round(elapsed, 1), "slow_beats": slow_count}
                    row.update(m)
                    rows.append(row)
                    print(f"  ✓ {tag}: AR@500={m['AR@500']:5.1f}% MedAE={m['MedAE_ms']:6.0f}ms MaxAE={m['MaxAE_ms']:6.0f}ms LR={m['LR500_K5']:5.1f}% ({elapsed:.0f}s)", flush=True)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  FAIL {tag}: {e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "metrics_dixon.csv", index=False)
    print(f"\nSaved {len(df)} runs to {OUT_ROOT}/metrics_dixon.csv")
    if len(df):
        print("\nPer-piece per-condition mean:")
        agg = df.groupby(["piece","condition"])[["AR@500","MedAE_ms","MaxAE_ms","LR500_K5"]].mean().round(1)
        print(agg)


if __name__ == "__main__":
    main()
