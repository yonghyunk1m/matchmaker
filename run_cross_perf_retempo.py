"""Cross-performer retempo proxy: simulate pass-to-pass variation.

For each pair (A, B):
- Estimate global ρ_A from performer A's audio (Pass-1)
- Re-render reference at ρ_A · T_0
- Run Dixon+CQT alignment on performer B's audio (proxy "Pass-2")
- Measure AR@500

Compare to:
- Same-performer retempo (B's own ρ_B applied to B): "fair rehearsal"
- No rehearsal (default reference, ρ=1): baseline

If cross-performer ρ_A still helps B vs baseline → robust to person-to-person variation
(proxy for real pass-to-pass interpretation drift).
"""
import os
import sys
import time

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


def estimate_rho(piece, performer):
    """Pass-1: estimate global ratio ρ from performer's audio (Dixon+STFT, default tempo)."""
    piece_dir = piece["asap_dir"]
    xml_path = os.path.join(piece_dir, "xml_score.musicxml")
    wav_path = os.path.join(piece_dir, f"{performer}.wav")
    if not os.path.exists(wav_path):
        return None

    initial_tempo = _get_xml_tempo(xml_path)
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
        return float(wp0[0][-1] / wp0[1][-1])
    return None


def run_pass2_with_rho(piece, performer, rho):
    """Pass-2: align performer's audio against reference re-rendered at rho*T_0."""
    piece_dir = piece["asap_dir"]
    xml_path = os.path.join(piece_dir, "xml_score.musicxml")
    wav_path = os.path.join(piece_dir, f"{performer}.wav")
    if not os.path.exists(wav_path):
        return None

    initial_tempo = _get_xml_tempo(xml_path)
    tempo_for_eval = initial_tempo * rho
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

    # Use CQT chroma
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
    ar500 = float(np.sum(err_ms <= 500) / len(err_ms) * 100)
    return ar500


# Op.23 (Chopin Ballade 1)
PIECE = next(p for p in PIECES if p["id"] == "chopin_ballade1")
# Use 6 of the 7 performers (drop MunA19M as in main paper)
PERFORMERS = [p for p in PIECE["performers"] if p != "MunA19M"]


def main():
    # Step 1: estimate ρ for each performer (Pass-1)
    print("=== Step 1: Estimate ρ for each performer ===", flush=True)
    rhos = {}
    for perf in PERFORMERS:
        t0 = time.time()
        rho = estimate_rho(PIECE, perf)
        elapsed = time.time() - t0
        if rho is None:
            print(f"  {perf}: FAILED (no audio)", flush=True)
            continue
        rhos[perf] = rho
        print(f"  {perf}: ρ={rho:.4f} ({elapsed:.0f}s)", flush=True)

    if len(rhos) < 2:
        print("Not enough performers with valid ρ.", flush=True)
        return

    # Step 2: cross-performer pairs (each performer paired with one other in cyclic order)
    print("\n=== Step 2: Cross-performer Pass-2 with ρ_A on B ===", flush=True)
    perfs = list(rhos.keys())
    rows = []
    for i, b in enumerate(perfs):
        a = perfs[(i + 1) % len(perfs)]  # cycle: each B uses next A's rho
        rho_a = rhos[a]
        rho_b = rhos[b]
        t0 = time.time()
        ar_cross = run_pass2_with_rho(PIECE, b, rho_a)
        elapsed = time.time() - t0
        if ar_cross is None:
            print(f"  {a}→{b}: FAILED", flush=True)
            continue
        print(f"  {a}→{b}: ρ_A={rho_a:.3f}, ρ_B={rho_b:.3f}, AR@500(B with ρ_A)={ar_cross:.1f}% ({elapsed:.0f}s)",
              flush=True)
        rows.append({
            "pass1_perf": a, "pass2_perf": b,
            "rho_pass1": rho_a, "rho_pass2_own": rho_b,
            "ar500_cross": ar_cross,
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/cross_perf_retempo.csv", index=False)
    print(f"\nSaved {len(df)} cross-pairs to results/cross_perf_retempo.csv")
    print(f"Mean cross-performer AR@500: {df['ar500_cross'].mean():.2f}%")


if __name__ == "__main__":
    main()
