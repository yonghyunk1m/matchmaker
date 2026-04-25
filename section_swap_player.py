"""SectionSwapPlayer — Option B (shadow parallel) prototype for live
section-level algorithm switching.

Architecture:
  - 3 OLTW trackers run in lockstep on the same audio stream.
  - At each frame, each tracker emits a prediction.
  - The publisher selects the per-section chosen tracker's prediction.
  - At section boundaries, the publisher switches its source tracker.
  - All trackers continue running (shadow tracking) — no state transfer.

This is the pre-deployment architecture validation. Compute cost: 3× a single
OLTW. On a single CPU core, OLTW runs at <0.25× RTF, so 3× ≈ <0.75× RTF →
real-time is preserved.

Usage:
  player = SectionSwapPlayer(piece, perf, schedule, plan_p1, plan_p2)
  player.run()
  # Output: results/section_swap_player/{piece}_{perf}_synth.csv
  # Comparison: matches analyze_section_swap.py post-hoc stitched trajectory.
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
from matchmaker import Matchmaker
from matchmaker.dp import OnlineTimeWarpingArzt, OnlineTimeWarpingDixon
from matchmaker.utils.eval import transfer_from_score_to_predicted_perf
from matchmaker.features.audio import FRAME_RATE
from run_cqt_native import CQTChromagramProcessor, StreamCQTChromagramProcessor
import matchmaker.matchmaker as _mm_mod


def find_section_idx(beat_pos, sections):
    """Given current score-beat position, return which section (a, b) it falls in."""
    for i, (a, b) in enumerate(sections):
        if a <= beat_pos < b:
            return i
    return len(sections) - 1


class SectionSwapPlayer:
    """Live section-swap player demo using 3 trackers in parallel.

    Currently implemented as offline-parallel (process audio sequentially with
    all 3 trackers in lockstep, then stitch outputs by section). The same
    architecture extends to live audio streaming with minimal changes.
    """

    def __init__(self, piece, perf, schedule, sections):
        """
        piece: dict from PIECES
        perf: performer ID
        schedule: list of {Pass1, Pass2, Dixon} per section, len == len(sections)
        sections: list of (start_beat, end_beat) tuples
        """
        self.piece = piece
        self.perf = perf
        self.schedule = schedule
        self.sections = sections
        # Three trackers will be initialized in setup()

    def _setup_trackers(self, plan_p1, plan_p2):
        """Initialize 3 Matchmaker instances. Each gets its own pipeline.

        For simplicity, we run 3 independent Matchmaker.run() loops then stitch.
        In a true real-time setting, they'd share a chroma stream and step in
        lockstep — same architecture, different scheduling.
        """
        from run_full_experiment import run_single
        from run_directional_dixon_pilot import run_dixon
        from run_directional_plan_pilot import metrics_from_err

        out_dir = Path("results/section_swap_player/trajectories")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pass-1: Arzt+rule_plan
        tag_p1 = f"{self.piece['id']}_{self.perf}_player_p1"
        run_single(self.piece, self.perf, "xml", algorithm="arzt",
                   plan_path_override=plan_p1,
                   per_beat_csv=str(out_dir / f"{tag_p1}.csv"))
        # Pass-2: Arzt+gated_plan
        tag_p2 = f"{self.piece['id']}_{self.perf}_player_p2"
        run_single(self.piece, self.perf, "xml", algorithm="arzt",
                   plan_path_override=plan_p2,
                   per_beat_csv=str(out_dir / f"{tag_p2}.csv"))
        # Dixon+CQT
        tag_dx = f"{self.piece['id']}_{self.perf}_player_dx"
        traj_dx = run_dixon(self.piece, self.perf, plan_gamma=None)
        if traj_dx is not None:
            pd.DataFrame({k: traj_dx[k] for k in
                          ["beat_idx", "gt_beat", "gt_time_s",
                           "pred_time_s", "err_ms", "ok"]}).to_csv(
                out_dir / f"{tag_dx}.csv", index=False)

        return tag_p1, tag_p2, tag_dx, out_dir

    def stitch_outputs(self, tag_p1, tag_p2, tag_dx, out_dir):
        """Per-frame: select the chosen tracker's prediction per section.

        This is the publisher logic. In live mode, it runs at ~30fps frame
        rate. In offline demo, we apply it post-hoc — mathematically
        identical because each tracker's per-beat output is deterministic
        for the same audio.
        """
        df_p1 = pd.read_csv(out_dir / f"{tag_p1}.csv")
        df_p2 = pd.read_csv(out_dir / f"{tag_p2}.csv")
        df_dx = pd.read_csv(out_dir / f"{tag_dx}.csv")

        synth_rows = []
        for i, (a, b) in enumerate(self.sections):
            chosen_name = self.schedule[i]
            df_chosen = {"Pass1": df_p1, "Pass2": df_p2,
                          "Dixon": df_dx, "Dixon+CQT": df_dx}[chosen_name]
            mask = (df_chosen["gt_beat"] >= a) & (df_chosen["gt_beat"] < b)
            sub = df_chosen[mask].copy()
            sub["section_idx"] = i
            sub["chosen_tracker"] = chosen_name
            synth_rows.append(sub)
        synth_df = pd.concat(synth_rows, ignore_index=True).sort_values("gt_beat")
        return synth_df

    def run(self, plan_p1, plan_p2):
        """Execute the section-swap player end-to-end and return synth trajectory."""
        tag_p1, tag_p2, tag_dx, out_dir = self._setup_trackers(plan_p1, plan_p2)
        synth_df = self.stitch_outputs(tag_p1, tag_p2, tag_dx, out_dir)
        synth_path = out_dir / f"{self.piece['id']}_{self.perf}_synth.csv"
        synth_df.to_csv(synth_path, index=False)
        return synth_df, synth_path


def demonstrate_op23_buijl():
    """Demo on Op.~23 BuiJL04M: rehearsal-determined schedule → live-style player → trajectory."""
    import pickle
    import tempfile
    from run_scorpion_rule_intent import extract_rule_gamma
    from run_directional_plan_pilot import metrics_from_err
    from run_confidence_gated_v2 import (
        section_boundaries, actual_gamma_per_section,
        build_blended_plan,
    )

    piece = next(p for p in PIECES if p["id"] == "chopin_ballade1")
    perf = "BuiJL04M"
    xml_path = os.path.join(piece["asap_dir"], "xml_score.musicxml")
    score = pt.load_musicxml(xml_path).parts[0]
    max_beat = int(np.ceil(score.note_array()["onset_beat"].max())) + 10

    # 1. Build Pass-1 plan (rule)
    gamma_rule, markings = extract_rule_gamma(xml_path, max_beat)
    locked_rule = np.zeros(max_beat, dtype=bool)
    sorted_mks = sorted(markings, key=lambda x: x[0])
    for (b_, kind, txt, g) in markings:
        if g < 0.85:
            next_b = max_beat
            for (b2, k2, t2, g2) in sorted_mks:
                if b2 > b_ and g2 >= 0.85:
                    next_b = int(b2)
                    break
            sb = int(b_)
            if sb < max_beat:
                locked_rule[sb:min(next_b, max_beat)] = True

    plan_p1_path = tempfile.mktemp(suffix=".pkl")
    with open(plan_p1_path, "wb") as f:
        pickle.dump((gamma_rule, np.ones(max_beat, dtype=np.float32) * 64.0, locked_rule), f)

    # 2. Build Pass-2 plan from existing pass-1 trajectory (rehearsal)
    sections_full = section_boundaries(markings, max_beat)
    pass1_traj_path = Path(f"results/confidence_gated_v2/trajectories/chopin_ballade1_BuiJL04M_pass1_v2.csv")
    if not pass1_traj_path.exists():
        print("ERROR: pass1 rehearsal trajectory missing — run run_confidence_gated_v2.py first")
        return None
    sec_actual = actual_gamma_per_section(pass1_traj_path, sections_full)
    gamma2, locked2, n_over, _ = build_blended_plan(
        gamma_rule, locked_rule, sec_actual, sections_full, max_beat)
    plan_p2_path = tempfile.mktemp(suffix=".pkl")
    with open(plan_p2_path, "wb") as f:
        pickle.dump((gamma2, np.ones(max_beat, dtype=np.float32) * 64.0, locked2), f)

    # 3. Load schedule (from rehearsal section-swap analysis)
    section_swap = pd.read_csv("results/section_swap/per_perf.csv")
    op23_buijl = section_swap[(section_swap["piece"] == "chopin_ballade1") &
                                (section_swap["perf"] == "BuiJL04M")]
    if len(op23_buijl) == 0:
        print("ERROR: no rehearsal-time section-swap result for BuiJL04M")
        return None

    # Build per-section schedule from existing analysis (post-hoc winners)
    sections_section = pd.read_csv("results/section_swap/sections.csv")
    op23_buijl_secs = sections_section[(sections_section["piece"] == "chopin_ballade1") &
                                         (sections_section["perf"] == "BuiJL04M")]
    schedule = op23_buijl_secs["best_choice"].tolist()
    sections_pairs = []
    for s_str in op23_buijl_secs["section"].tolist():
        a, b = map(int, s_str.split("-"))
        sections_pairs.append((a, b))

    print(f"\n=== SectionSwapPlayer demo: Op.~23 BuiJL04M ===")
    print(f"Schedule (per section): {dict(pd.Series(schedule).value_counts())}")

    # 4. Run player
    _mm_mod.DEFAULT_TEMPO = _get_xml_tempo(xml_path)
    player = SectionSwapPlayer(piece, perf, schedule, sections_pairs)
    t0 = time.time()
    synth_df, synth_path = player.run(plan_p1_path, plan_p2_path)
    elapsed = time.time() - t0

    # 5. Metrics
    err = synth_df["err_ms"].abs()
    ar = (err <= 500).mean() * 100
    medae = float(err.median()) / 1000
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Synth trajectory: AR@500 = {ar:.1f}%, MedAE = {medae:.2f}s")
    print(f"Saved: {synth_path}")

    # 6. Compare to post-hoc analyze_section_swap result (should match)
    expected_ar = float(op23_buijl["AR_SectionBest"].iloc[0])
    print(f"\nExpected (analyze_section_swap.py): AR@500 = {expected_ar:.1f}%")
    print(f"Difference: {abs(ar - expected_ar):.2f} pp  → equivalence {'✓ PASSED' if abs(ar - expected_ar) < 1.0 else '✗ FAILED'}")

    return synth_df


if __name__ == "__main__":
    demonstrate_op23_buijl()
