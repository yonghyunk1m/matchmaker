"""Run Dixon+CQT baseline for pieces that have Pass1/Pass2 trajectories
but no Dixon+CQT trajectory yet — needed for section-swap analysis.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_experiment import PIECES
from run_directional_dixon_pilot import run_dixon
from run_directional_plan_pilot import metrics_from_err

OUT_DIR = Path("results/hybrid_full/trajectories")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pieces with Pass1/Pass2 but no Dixon yet
TARGET_PIECES = {
    "chopin_etude_25_11", "schubert_impromptu_3",
    "chopin_scherzo_31", "bach_prelude_848",
}


def main():
    rows = []
    for piece in PIECES:
        pid = piece["id"]
        if pid not in TARGET_PIECES:
            continue
        for perf in piece["performers"]:
            wav = os.path.join(piece["asap_dir"], f"{perf}.wav")
            if not os.path.exists(wav):
                continue
            tag = f"{pid}_{perf}_dixon_cqt"
            out_csv = OUT_DIR / f"{tag}.csv"
            if out_csv.exists():
                print(f"  ◇ {tag} already exists, skip")
                continue
            t0 = time.time()
            try:
                traj = run_dixon(piece, perf, plan_gamma=None)
                if traj is None:
                    continue
                df = pd.DataFrame({k: traj[k] for k in
                                   ["beat_idx", "gt_beat", "gt_time_s",
                                    "pred_time_s", "err_ms", "ok"]})
                df.to_csv(out_csv, index=False)
                m = metrics_from_err(traj["err_ms"])
                el = time.time() - t0
                print(f"  ✓ {tag}: AR@500={m['AR@500']:5.1f}% MedAE={m['MedAE_ms']:6.0f}ms ({el:.0f}s)", flush=True)
                rows.append({"piece": pid, "perf": perf, **m})
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL {tag}: {e}", flush=True)
    pd.DataFrame(rows).to_csv("results/dixon_cqt_batch.csv", index=False)
    print(f"\nSaved {len(rows)} rows")


if __name__ == "__main__":
    main()
