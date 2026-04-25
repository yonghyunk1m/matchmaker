"""Cross-piece confidence-gated rehearsal: Op.~38, Schubert (cascade-likely),
Bach (control). Enables section-swap analysis on multiple pieces.
"""
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import partitura as pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_experiment import PIECES, run_single
from run_directional_plan_pilot import metrics_from_err
from run_scorpion_rule_intent import extract_rule_gamma
from run_confidence_gated_v2 import (
    section_boundaries, actual_gamma_per_section,
    build_blended_plan, CONFIDENCE_TOL, MODERATO_BPS
)

OUT_ROOT = Path("results/confidence_gated_v2")
OUT_TRAJ = OUT_ROOT / "trajectories"
OUT_TRAJ.mkdir(parents=True, exist_ok=True)

PILOT = [("chopin_ballade_2", 3), ("schubert_impromptu_3", 3), ("bach_prelude_848", 2)]


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
        sections = section_boundaries(markings, max_beat)
        print(f"\n{pid}: max_beat={max_beat}, sections={len(sections)}, "
              f"locked={int(locked_rule.sum())}", flush=True)

        plan_p1 = tempfile.mktemp(suffix=".pkl")
        with open(plan_p1, "wb") as f:
            pickle.dump((gamma_rule, np.ones(max_beat, dtype=np.float32) * 64.0, locked_rule), f)

        for perf in perfs:
            wav = os.path.join(piece["asap_dir"], f"{perf}.wav")
            if not os.path.exists(wav):
                continue
            tag1 = f"{pid}_{perf}_pass1_v2"
            tag2 = f"{pid}_{perf}_pass2_v2"
            t0 = time.time()
            try:
                run_single(piece, perf, "xml", algorithm="arzt",
                           plan_path_override=plan_p1,
                           per_beat_csv=str(OUT_TRAJ / f"{tag1}.csv"))
                df1 = pd.read_csv(OUT_TRAJ / f"{tag1}.csv")
                m1 = metrics_from_err(df1["err_ms"].values)
                print(f"  ✓ {tag1}: AR@500={m1['AR@500']:5.1f}%  ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                continue

            sec_actual = actual_gamma_per_section(OUT_TRAJ / f"{tag1}.csv", sections)
            gamma2, locked2, n_over, _ = build_blended_plan(
                gamma_rule, locked_rule, sec_actual, sections, max_beat)
            print(f"     → {n_over} overrides")
            plan_p2 = tempfile.mktemp(suffix=".pkl")
            with open(plan_p2, "wb") as f:
                pickle.dump((gamma2, np.ones(max_beat, dtype=np.float32) * 64.0, locked2), f)

            t0 = time.time()
            try:
                run_single(piece, perf, "xml", algorithm="arzt",
                           plan_path_override=plan_p2,
                           per_beat_csv=str(OUT_TRAJ / f"{tag2}.csv"))
                df2 = pd.read_csv(OUT_TRAJ / f"{tag2}.csv")
                m2 = metrics_from_err(df2["err_ms"].values)
                d = m2["AR@500"] - m1["AR@500"]
                print(f"  ✓ {tag2}: AR@500={m2['AR@500']:5.1f}%  Δ={d:+5.1f}pp  ({time.time()-t0:.0f}s)", flush=True)
                rows.append({"piece": pid, "perf": perf, "n_overrides": n_over,
                             "AR@500_p1": m1["AR@500"], "AR@500_p2": m2["AR@500"],
                             "delta_AR500": d})
            except Exception as e:
                import traceback; traceback.print_exc()
            os.unlink(plan_p2)
        if os.path.exists(plan_p1):
            os.unlink(plan_p1)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "cross_piece_pass2.csv", index=False)
    print(f"\nSaved {len(df)} cross-piece pass2 rows")


if __name__ == "__main__":
    main()
