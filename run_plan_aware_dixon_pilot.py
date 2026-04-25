"""Plan-aware Dixon: cost-surface bias in slow-marked regions.

Mechanism (constraint, not prediction):
- Inherit OnlineTimeWarpingDixon (path-length normalization, no forward bias)
- Override select_candidate: in slow-marked beats, bias norm_x_edge UP
  → Dixon prefers norm_y_edge (advance input, hold reference) → "wait" behavior
- No audio re-render. No prediction of actual performer tempo.
- Bit-identical to baseline outside slow regions.

Key difference from Path B:
  Path B: PREDICT performer's slow tempo, render reference accordingly (brittle)
  This:   CONSTRAIN tracker to wait in slow regions (robust)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import partitura as pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_experiment import PIECES, load_gt, SAMPLE_RATE, _get_xml_tempo, _build_score_times_for_beats
from run_directional_plan_pilot import build_directional_plan, metrics_from_err, PILOT
from matchmaker import Matchmaker
from matchmaker.dp import OnlineTimeWarpingDixon
from matchmaker.dp.oltw_dixon import Direction
from matchmaker.utils.eval import transfer_from_score_to_predicted_perf
from matchmaker.features.audio import FRAME_RATE
from run_cqt_native import CQTChromagramProcessor, StreamCQTChromagramProcessor
import matchmaker.matchmaker as _mm_mod
import librosa


class PlanAwareDixon(OnlineTimeWarpingDixon):
    """Dixon with plan-aware cost bias in slow-marked regions."""

    def __init__(self, *args, plan_gamma=None, ref_frames_per_beat=None,
                 penalty_strength=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan_gamma = plan_gamma   # array of γ per beat
        self.ref_frames_per_beat = ref_frames_per_beat   # for beat ↔ frame map
        self.penalty_strength = penalty_strength

    def _ref_frame_to_beat(self, ref_frame):
        if self.ref_frames_per_beat is None:
            return None
        return int(ref_frame / self.ref_frames_per_beat)

    def select_candidate(self):
        norm_x_edge = self.acc_dist_matrix[-1, :] / self.acc_len_matrix[-1, :]
        norm_y_edge = self.acc_dist_matrix[:, -1] / self.acc_len_matrix[:, -1]

        # PLAN INJECTION
        if self.plan_gamma is not None and self.ref_frames_per_beat is not None:
            beat_idx = self._ref_frame_to_beat(self.ref_pointer)
            if beat_idx is not None and 0 <= beat_idx < len(self.plan_gamma):
                gamma = float(self.plan_gamma[beat_idx])
                if gamma < 1.0:
                    # In slow region: penalize ref advancement (norm_x_edge)
                    # makes "wait" (norm_y_edge) relatively cheaper
                    penalty = self.penalty_strength * (1.0 - gamma)  # 0.4 → 0.6 penalty
                    norm_x_edge = norm_x_edge * (1.0 + penalty)

        cat = np.concatenate((norm_x_edge, norm_y_edge))
        min_idx = np.argmin(cat)
        offset = self.offset()
        if min_idx <= len(norm_x_edge):
            self.candidate = np.array([self.ref_pointer - offset[0], min_idx])
        else:
            self.candidate = np.array(
                [min_idx - len(norm_x_edge), self.input_pointer - offset[1]]
            )


OUT_ROOT = Path("results/directional_plan")
OUT_TRAJ = OUT_ROOT / "trajectories"


def run_plan_aware_dixon(piece, performer, plan_gamma=None, penalty_strength=1.0):
    piece_dir = piece["asap_dir"]
    xml_path = os.path.join(piece_dir, "xml_score.musicxml")
    wav_path = os.path.join(piece_dir, f"{performer}.wav")
    if not os.path.exists(wav_path):
        return None

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

    # Compute frames per beat
    ref_dur = len(mm.score_audio) / SAMPLE_RATE
    n_ref_frames = mm.reference_features.shape[0]
    sp_bmin = max(na["onset_beat"].min(), 0.0)
    sp_bmax = na["onset_beat"].max()
    n_beats = sp_bmax - sp_bmin
    ref_frames_per_beat = n_ref_frames / max(n_beats, 1)

    if plan_gamma is not None:
        mm.score_follower = PlanAwareDixon(
            reference_features=mm.reference_features, queue=mm.stream.queue,
            plan_gamma=plan_gamma, ref_frames_per_beat=ref_frames_per_beat,
            penalty_strength=penalty_strength)
    else:
        mm.score_follower = OnlineTimeWarpingDixon(
            reference_features=mm.reference_features, queue=mm.stream.queue)

    for _ in mm.run(verbose=False):
        pass
    wp = np.array(mm.score_follower.warping_path)

    gt_pt, gt_bt = load_gt(piece_dir, performer)
    lut_b = np.linspace(sp_bmin, sp_bmax, 2000)
    lut_t = _build_score_times_for_beats(mm.score_part, lut_b, initial_tempo)
    gt_beats_scaled = gt_bt * (bt / 4)
    score_annots = np.interp(gt_beats_scaled, lut_b, lut_t) - trim
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
            continue
        perfs = piece["performers"][:n_perf]
        xml_path = os.path.join(piece["asap_dir"], "xml_score.musicxml")
        score = pt.load_musicxml(xml_path).parts[0]
        max_beat = int(np.ceil(score.note_array()["onset_beat"].max())) + 10
        gamma, _, _, n_slow = build_directional_plan(xml_path, max_beat, slow_gamma=0.4)
        slow_count = int(np.sum(gamma != 1.0))
        print(f"\n{pid}: slow_beats={slow_count}", flush=True)

        for perf in perfs:
            tag = f"{pid}_{perf}_C8_dixon_planaware"
            t0 = time.time()
            try:
                traj = run_plan_aware_dixon(piece, perf, plan_gamma=gamma, penalty_strength=1.0)
                if traj is None:
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
                row = {"piece": pid, "perf": perf, "condition": "C8_dixon_planaware",
                       "slow_beats": slow_count, "elapsed_s": round(elapsed, 1)}
                row.update(m)
                rows.append(row)
                print(f"  ✓ {tag}: AR@500={m['AR@500']:5.1f}% MedAE={m['MedAE_ms']:6.0f}ms MaxAE={m['MaxAE_ms']:6.0f}ms ({elapsed:.0f}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL {tag}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "metrics_dixon_planaware.csv", index=False)
    print(f"\nSaved {len(df)} runs")


if __name__ == "__main__":
    main()
