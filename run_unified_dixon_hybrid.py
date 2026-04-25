"""Unified OLTW-level hybrid: Dixon + step-size cap from PerformancePlan.

Mechanism (single tracker, no algorithm switching):
  - Dixon's path-length normalization (avoids Arzt's forward bias on cascades)
  - + per-beat δ_cap from γ(b) bound to max_run_count (Dixon's consecutive-step limit)
  - Slow region (γ<1):  max_run_count = floor(5γ)  -> tighter cap, prevents skip-ahead
  - Default (γ≥1):      max_run_count = 30 (Dixon's standard)

This is the "single algorithm, both roles" the user asked for: Dixon's robust
normalization + Arzt-plan's structural cap, in one tracker.

Pilot: Op.~23 n=7 + Op.~38 (control: slow_beats=0 → cap inactive → identical to baseline).
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

from run_full_experiment import PIECES, load_gt, SAMPLE_RATE, _get_xml_tempo, _build_score_times_for_beats
from run_directional_plan_pilot import metrics_from_err
from run_scorpion_rule_intent import extract_rule_gamma
from matchmaker import Matchmaker
from matchmaker.dp import OnlineTimeWarpingDixon
import matchmaker.matchmaker as _mm_mod
from matchmaker.utils.eval import transfer_from_score_to_predicted_perf
from matchmaker.features.audio import FRAME_RATE
from run_cqt_native import CQTChromagramProcessor, StreamCQTChromagramProcessor


class UnifiedDixonHybrid(OnlineTimeWarpingDixon):
    """Dixon with per-beat max_run_count cap from PerformancePlan."""

    def __init__(self, *args, plan_gamma=None, ref_frames_per_beat=None,
                 default_max_run=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan_gamma = plan_gamma
        self.ref_frames_per_beat = ref_frames_per_beat
        self.default_max_run = default_max_run
        self._cap_history = []

    def _current_beat(self):
        if self.ref_frames_per_beat is None or self.ref_frames_per_beat <= 0:
            return -1
        return int(self.ref_pointer / self.ref_frames_per_beat)

    def select_next_direction(self):
        # Adapt max_run_count from plan γ at current beat
        if self.plan_gamma is not None:
            beat = self._current_beat()
            if 0 <= beat < len(self.plan_gamma):
                gamma = float(self.plan_gamma[beat])
                if gamma < 1.0:
                    # Tighter cap → fewer consecutive REF before forced toggle
                    self.max_run_count = max(1, int(5 * gamma))
                else:
                    self.max_run_count = self.default_max_run
                self._cap_history.append((beat, gamma, self.max_run_count))
        return super().select_next_direction()


OUT_ROOT = Path("results/unified_dixon_hybrid")
OUT_TRAJ = OUT_ROOT / "trajectories"
OUT_TRAJ.mkdir(parents=True, exist_ok=True)


def run_unified(piece, performer, plan_gamma=None):
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

    ref_dur = len(mm.score_audio) / SAMPLE_RATE
    n_ref_frames = mm.reference_features.shape[0]
    sp_bmin = max(na["onset_beat"].min(), 0.0)
    sp_bmax = na["onset_beat"].max()
    n_beats = sp_bmax - sp_bmin
    ref_frames_per_beat = n_ref_frames / max(n_beats, 1)

    if plan_gamma is not None:
        mm.score_follower = UnifiedDixonHybrid(
            reference_features=mm.reference_features, queue=mm.stream.queue,
            plan_gamma=plan_gamma, ref_frames_per_beat=ref_frames_per_beat)
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
    PILOT = [("chopin_ballade1", None), ("chopin_ballade_2", 2), ("bach_prelude_848", 2)]
    rows = []
    for pid, n_perf in PILOT:
        try:
            piece = next(p for p in PIECES if p["id"] == pid)
        except StopIteration:
            continue
        perfs = piece["performers"] if n_perf is None else piece["performers"][:n_perf]
        xml_path = os.path.join(piece["asap_dir"], "xml_score.musicxml")
        score = pt.load_musicxml(xml_path).parts[0]
        max_beat = int(np.ceil(score.note_array()["onset_beat"].max())) + 10
        gamma_rule, _ = extract_rule_gamma(xml_path, max_beat)
        slow_beats = int(np.sum(gamma_rule < 0.85))
        print(f"\n{pid}: slow_beats(γ<0.85)={slow_beats}", flush=True)

        for perf in perfs:
            tag_h = f"{pid}_{perf}_unified_hybrid"
            tag_b = f"{pid}_{perf}_dixon_baseline_cqt"

            t0 = time.time()
            try:
                traj_h = run_unified(piece, perf, plan_gamma=gamma_rule)
                if traj_h is None:
                    continue
                pd.DataFrame({k: traj_h[k] for k in
                              ["beat_idx", "gt_beat", "gt_time_s", "pred_time_s", "err_ms", "ok"]}).to_csv(
                    OUT_TRAJ / f"{tag_h}.csv", index=False)
                m_h = metrics_from_err(traj_h["err_ms"])
                el_h = time.time() - t0
                print(f"  ✓ U {tag_h}: AR@500={m_h['AR@500']:5.1f}% MedAE={m_h['MedAE_ms']:6.0f}ms ({el_h:.0f}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL U {tag_h}: {e}")
                continue

            t0 = time.time()
            try:
                traj_b = run_unified(piece, perf, plan_gamma=None)
                if traj_b is None:
                    continue
                pd.DataFrame({k: traj_b[k] for k in
                              ["beat_idx", "gt_beat", "gt_time_s", "pred_time_s", "err_ms", "ok"]}).to_csv(
                    OUT_TRAJ / f"{tag_b}.csv", index=False)
                m_b = metrics_from_err(traj_b["err_ms"])
                el_b = time.time() - t0
                d = m_h["AR@500"] - m_b["AR@500"]
                print(f"  ✓ B {tag_b}:    AR@500={m_b['AR@500']:5.1f}% MedAE={m_b['MedAE_ms']:6.0f}ms  Δ={d:+5.1f}pp ({el_b:.0f}s)", flush=True)
                rows.append({"piece": pid, "perf": perf, "slow_beats": slow_beats,
                             "AR500_unified": m_h["AR@500"], "AR500_baseline": m_b["AR@500"],
                             "delta": d,
                             "MedAE_unified": m_h["MedAE_ms"], "MedAE_baseline": m_b["MedAE_ms"]})
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL B {tag_b}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "metrics.csv", index=False)
    print(f"\nSaved {len(df)} pairs")
    if len(df):
        print(f"\nMean Δ unified−baseline: {df['delta'].mean():+.2f}pp")
        print(f"Improvements (>+0.5): {int((df['delta']>0.5).sum())}/{len(df)}")
        print(f"Regressions (<-0.5):  {int((df['delta']<-0.5).sum())}/{len(df)}")
        for _, r in df.iterrows():
            print(f"  {r['piece']:18s} {r['perf']:18s} U {r['AR500_unified']:5.1f}  B {r['AR500_baseline']:5.1f}  Δ {r['delta']:+5.1f}pp")


if __name__ == "__main__":
    main()
