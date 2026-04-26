"""Dixon+CQT+retempo per-beat trajectory generator for server 2.

Batch 4: fill the 6 pieces missing from cooperative pool so the 4-tracker pool
is symmetric across all 10 pieces in section_swap. Cached files (batch 3) are
skipped automatically — this run only does the new pieces.

Output: results/dixon_cqt_retempo/{piece}_{perf}_C2_global_retempo.csv

Server 2 instructions:
  cd ~/matchmaker
  git pull origin develop
  python3 run_retempo_batch_server2.py 2>&1 | tee /tmp/retempo_b4.log
  # then push results to a new branch:
  git checkout -b retempo-trajectories-server2-batch4
  git add results/dixon_cqt_retempo/*.csv
  git commit -m "Server 2 batch 4: retempo trajectories for 6 pieces"
  git push -u origin retempo-trajectories-server2-batch4
"""
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import partitura as pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_experiment import PIECES, load_gt, SAMPLE_RATE, _get_xml_tempo, _build_score_times_for_beats
from run_directional_plan_pilot import metrics_from_err
from matchmaker import Matchmaker
from matchmaker.dp import OnlineTimeWarpingDixon
import matchmaker.matchmaker as _mm_mod
from matchmaker.utils.eval import transfer_from_score_to_predicted_perf
from matchmaker.features.audio import FRAME_RATE
from run_cqt_native import CQTChromagramProcessor, StreamCQTChromagramProcessor

OUT_DIR = Path("results/dixon_cqt_retempo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASSIGNED = {
    "chopin_ballade1", "chopin_ballade_2", "chopin_barcarolle",
    "schubert_impromptu_3", "chopin_etude_25_11", "bach_prelude_848",
}


def run_pass1_get_global_ratio(piece, performer):
    """Pass 1: Dixon+CQT baseline → derive global ρ = ref_dur / perf_dur from warping path."""
    piece_dir = piece["asap_dir"]
    xml_path = os.path.join(piece_dir, "xml_score.musicxml")
    wav_path = os.path.join(piece_dir, f"{performer}.wav")
    initial_tempo = _get_xml_tempo(xml_path)
    _mm_mod.DEFAULT_TEMPO = initial_tempo
    mm = Matchmaker(score_file=xml_path, performance_file=wav_path,
                    input_type="audio", wait=False)
    na = mm.score_part.note_array()
    first_beat = na["onset_beat"].min()
    bt = mm.score_part.time_sigs[0].beat_type if mm.score_part.time_sigs else 4
    trim = max(first_beat, 0) * (60.0 / initial_tempo)
    ts = int(trim * SAMPLE_RATE)
    if ts > 0:
        mm.score_audio = mm.score_audio[ts:]
    mm.processor = CQTChromagramProcessor(sample_rate=SAMPLE_RATE)
    mm.reference_features = mm.processor(mm.score_audio)
    perf_y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    mm.stream.processor = StreamCQTChromagramProcessor(perf_y, sample_rate=SAMPLE_RATE)
    mm.score_follower = OnlineTimeWarpingDixon(
        reference_features=mm.reference_features, queue=mm.stream.queue)
    for _ in mm.run(verbose=False):
        pass
    wp = np.array(mm.score_follower.warping_path)
    ref_dur = len(mm.score_audio) / SAMPLE_RATE
    last_perf_t = wp[1, -1] / FRAME_RATE if len(wp[1]) > 0 else len(perf_y) / SAMPLE_RATE
    rho = ref_dur / max(last_perf_t, 1e-3)
    return rho, initial_tempo, trim


def run_pass2_with_retempo(piece, performer, rho, initial_tempo, trim):
    """Pass 2: re-render reference at rho * tempo, run Dixon+CQT."""
    piece_dir = piece["asap_dir"]
    xml_path = os.path.join(piece_dir, "xml_score.musicxml")
    wav_path = os.path.join(piece_dir, f"{performer}.wav")
    new_tempo = initial_tempo * rho
    _mm_mod.DEFAULT_TEMPO = new_tempo
    mm = Matchmaker(score_file=xml_path, performance_file=wav_path,
                    input_type="audio", wait=False)
    na = mm.score_part.note_array()
    first_beat = na["onset_beat"].min()
    bt = mm.score_part.time_sigs[0].beat_type if mm.score_part.time_sigs else 4
    new_trim = max(first_beat, 0) * (60.0 / new_tempo)
    ts = int(new_trim * SAMPLE_RATE)
    if ts > 0:
        mm.score_audio = mm.score_audio[ts:]
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
    lut_t = _build_score_times_for_beats(mm.score_part, lut_b, new_tempo)
    gt_beats_scaled = gt_bt * (bt / 4)
    score_annots = np.interp(gt_beats_scaled, lut_b, lut_t) - new_trim
    ref_dur = len(mm.score_audio) / SAMPLE_RATE
    score_annots = np.clip(score_annots, 0, ref_dur)
    pred = transfer_from_score_to_predicted_perf(wp, score_annots, FRAME_RATE)
    err_ms = np.where(np.isnan(pred), np.nan, np.abs(gt_pt - pred) * 1000.0)
    return {
        "beat_idx": np.arange(len(gt_pt)), "gt_beat": gt_bt,
        "gt_time_s": gt_pt, "pred_time_s": pred, "err_ms": err_ms,
        "ok": (~np.isnan(pred)).astype(int),
    }


def main():
    rows = []
    for piece in PIECES:
        if piece["id"] not in ASSIGNED:
            continue
        for perf in piece["performers"]:
            wav = os.path.join(piece["asap_dir"], f"{perf}.wav")
            if not os.path.exists(wav):
                continue
            tag = f"{piece['id']}_{perf}_C2_global_retempo"
            out_csv = OUT_DIR / f"{tag}.csv"
            if out_csv.exists():
                print(f"  ◇ {tag} cached, skip"); continue
            t0 = time.time()
            try:
                rho, initial_tempo, trim = run_pass1_get_global_ratio(piece, perf)
                print(f"  Pass-1 {piece['id']}/{perf}: ρ={rho:.3f}", flush=True)
                traj = run_pass2_with_retempo(piece, perf, rho, initial_tempo, trim)
                df = pd.DataFrame({k: traj[k] for k in
                                   ["beat_idx", "gt_beat", "gt_time_s",
                                    "pred_time_s", "err_ms", "ok"]})
                df.to_csv(out_csv, index=False)
                m = metrics_from_err(traj["err_ms"])
                el = time.time() - t0
                print(f"  ✓ {tag}: AR@500={m['AR@500']:5.1f}% MedAE={m['MedAE_ms']:6.0f}ms ρ={rho:.3f} ({el:.0f}s)", flush=True)
                rows.append({"piece": piece['id'], "perf": perf, "rho": rho, **m})
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL {tag}: {e}", flush=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nSaved {len(rows)} rows")


if __name__ == "__main__":
    main()
