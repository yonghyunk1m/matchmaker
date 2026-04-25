"""Dixon directional pilot with MILDER γ (0.7 instead of 0.4).

Hypothesis: γ=0.4 stretches reference 2.5× in slow regions — too aggressive,
performer doesn't slow that much. γ=0.7 stretches 1.43× — closer to actual
performer slow tempo (Op.~23 Meno mosso actual γ ≈ 0.74).

Conditions:
  C1: Dixon+CQT baseline
  C7: Dixon+CQT + directional Path B with γ_slow = 0.7
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import partitura as pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_experiment import PIECES
from run_directional_plan_pilot import build_directional_plan, metrics_from_err, PILOT
from run_directional_dixon_pilot import run_dixon

OUT_ROOT = Path("results/directional_plan")
OUT_TRAJ = OUT_ROOT / "trajectories"


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
        gamma_07, _, _, n_slow = build_directional_plan(xml_path, max_beat, slow_gamma=0.7)
        slow_count = int(np.sum(gamma_07 != 1.0))
        print(f"\n{pid}: slow_beats={slow_count} (γ=0.7)", flush=True)

        for perf in perfs:
            tag = f"{pid}_{perf}_C7_dixon_dir_mild"
            t0 = time.time()
            try:
                traj = run_dixon(piece, perf, plan_gamma=gamma_07)
                if traj is None:
                    print(f"  SKIP {tag}")
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
                row = {"piece": pid, "perf": perf, "condition": "C7_dixon_dir_mild",
                       "slow_gamma": 0.7, "slow_beats": slow_count, "elapsed_s": round(elapsed, 1)}
                row.update(m)
                rows.append(row)
                print(f"  ✓ {tag}: AR@500={m['AR@500']:5.1f}% MedAE={m['MedAE_ms']:6.0f}ms MaxAE={m['MaxAE_ms']:6.0f}ms ({elapsed:.0f}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL {tag}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "metrics_dixon_mild.csv", index=False)
    print(f"\nSaved {len(df)} runs")


if __name__ == "__main__":
    main()
