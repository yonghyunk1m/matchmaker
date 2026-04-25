"""LR pilot: compute Mean LR@500, K=5 on a representative 3-piece subset
across baseline (Arzt+STFT) and full stack (Dixon+CQT+retempo).

Modifies run_with_feature path to also return per-beat errors.

Output: results/lr_pilot.csv with columns piece, perf, condition, ar500, lr500_k5
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_experiment import PIECES, load_gt
from run_cqt_native import (
    run_with_feature, _get_xml_tempo, _build_score_times_for_beats,
    transfer_from_score_to_predicted_perf, SAMPLE_RATE, FRAME_RATE,
    CQTChromagramProcessor, StreamCQTChromagramProcessor,
    OnlineTimeWarpingDixon, OnlineTimeWarpingArzt, Matchmaker, _mm_mod,
)


def run_with_per_beat(piece, performer, algorithm='dixon', use_cqt=False, use_resynth=False):
    """Modified run_with_feature that returns (ar500, per_beat_err_ms)."""
    piece_dir = piece["asap_dir"]
    xml_path = os.path.join(piece_dir, "xml_score.musicxml")
    wav_path = os.path.join(piece_dir, f"{performer}.wav")
    if not os.path.exists(wav_path):
        return None, None

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

    if use_cqt:
        mm.processor = CQTChromagramProcessor(sample_rate=SAMPLE_RATE)
        mm.reference_features = mm.processor(mm.score_audio)
        import librosa as _lr
        perf_y, _ = _lr.load(wav_path, sr=SAMPLE_RATE)
        mm.stream.processor = StreamCQTChromagramProcessor(perf_y, sample_rate=SAMPLE_RATE)
    else:
        mm.reference_features = mm.processor(mm.score_audio)

    if algorithm == 'dixon':
        mm.score_follower = OnlineTimeWarpingDixon(
            reference_features=mm.reference_features, queue=mm.stream.queue)
    else:
        mm.score_follower = OnlineTimeWarpingArzt(
            reference_features=mm.reference_features, queue=mm.stream.queue,
            frame_rate=mm.frame_rate)

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
    # Per-beat err: NaN where prediction missing, else |gt - pred| in ms
    err_ms = np.where(np.isnan(pred), np.nan, np.abs(gt_pt - pred) * 1000.0)
    ar500 = float(np.sum(err_ms <= 500) / len(err_ms) * 100)
    return ar500, err_ms


def lr_at(err_ms, tau=500.0, K=5):
    """LR@(tau,K): fraction of beats b where b and preceding K-1 beats all <= tau.
    NaN counts as miss."""
    n = len(err_ms)
    if n < K:
        return 0.0
    within = (err_ms <= tau) & (~np.isnan(err_ms))
    locked = np.zeros(n, dtype=bool)
    for i in range(K - 1, n):
        locked[i] = within[i - K + 1: i + 1].all()
    return float(np.sum(locked) / n * 100)


# Representative subset
SUBSET_IDS = ["bach_prelude_848", "liszt_campanella", "chopin_ballade1"]
CONDITIONS = [
    ("arzt_stft", dict(algorithm="arzt", use_cqt=False, use_resynth=False)),
    ("dixon_cqt_retempo", dict(algorithm="dixon", use_cqt=True, use_resynth=True)),
]


def main():
    rows = []
    for pid in SUBSET_IDS:
        piece = next(p for p in PIECES if p["id"] == pid)
        for perf in piece["performers"]:
            for cname, cargs in CONDITIONS:
                t0 = time.time()
                try:
                    ar, err = run_with_per_beat(piece, perf, **cargs)
                    if ar is None:
                        continue
                    lr = lr_at(err, tau=500.0, K=5)
                    elapsed = time.time() - t0
                    print(f"{pid}/{perf}/{cname}: AR={ar:.1f}% LR={lr:.1f}% ({elapsed:.0f}s)", flush=True)
                    rows.append({"piece": pid, "perf": perf, "condition": cname,
                                 "ar500": ar, "lr500_k5": lr})
                except Exception as e:
                    print(f"FAIL {pid}/{perf}/{cname}: {e}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/lr_pilot.csv", index=False)
    print("\nMean per condition:")
    print(df.groupby("condition")[["ar500", "lr500_k5"]].mean().round(2))


if __name__ == "__main__":
    main()
