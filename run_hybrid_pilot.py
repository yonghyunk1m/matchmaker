"""Hybrid algorithm: auto-select Arzt+directional vs Dixon+CQT based on score.

Decision rule:
  slow_beats = count beats with directional γ < 1.0 (i.e., marked-slow regions)
  if slow_beats >= THRESHOLD (e.g., 50):
      → Arzt + directional plan via Path A (cascade-aware)
  else:
      → Dixon + CQT baseline (cascade-free, strong tracker)

Output:
  results/directional_plan/metrics_hybrid.csv
  results/directional_plan/trajectories/{piece}_{perf}_hybrid.csv
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
from run_directional_plan_pilot import build_directional_plan, metrics_from_err, PILOT
from run_directional_dixon_pilot import run_dixon

OUT_ROOT = Path("results/directional_plan")
OUT_TRAJ = OUT_ROOT / "trajectories"

THRESHOLD_SLOW_BEATS = 50  # below this → Dixon baseline


def select_algorithm(xml_path, max_beat):
    gamma, _, locked, n_slow = build_directional_plan(xml_path, max_beat, slow_gamma=0.4)
    slow_beats = int(np.sum(gamma != 1.0))
    if slow_beats >= THRESHOLD_SLOW_BEATS:
        return 'arzt_directional', gamma, locked, slow_beats
    return 'dixon_baseline', None, None, slow_beats


def main():
    rows = []
    for pid, n_perf in PILOT:
        try:
            piece = next(p for p in PIECES if p["id"] == pid)
        except StopIteration:
            print(f"  SKIP {pid}: not in PIECES")
            continue
        xml_path = os.path.join(piece["asap_dir"], "xml_score.musicxml")
        score = pt.load_musicxml(xml_path).parts[0]
        max_beat = int(np.ceil(score.note_array()["onset_beat"].max())) + 10
        algo, gamma, locked, slow_beats = select_algorithm(xml_path, max_beat)
        print(f"\n{pid}: slow_beats={slow_beats} → {algo}", flush=True)

        plan_path = None
        if algo == 'arzt_directional':
            dyn = np.ones(max_beat, dtype=np.float32) * 64.0
            plan_path = tempfile.mktemp(suffix=".pkl")
            with open(plan_path, "wb") as f:
                pickle.dump((gamma, dyn, locked), f)

        for perf in piece["performers"][:n_perf]:
            tag = f"{pid}_{perf}_hybrid"
            t0 = time.time()
            try:
                if algo == 'arzt_directional':
                    res = run_single(piece, perf, "xml", algorithm="arzt",
                                     plan_path_override=plan_path,
                                     per_beat_csv=str(OUT_TRAJ / f"{tag}.csv"))
                    df = pd.read_csv(OUT_TRAJ / f"{tag}.csv")
                    err = df["err_ms"].values
                else:
                    traj = run_dixon(piece, perf, plan_gamma=None)
                    if traj is None:
                        print(f"  SKIP {tag}: no audio")
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
                    err = traj["err_ms"]

                m = metrics_from_err(err)
                elapsed = time.time() - t0
                row = {"piece": pid, "perf": perf, "selected_algo": algo,
                       "slow_beats": slow_beats, "elapsed_s": round(elapsed, 1)}
                row.update(m)
                rows.append(row)
                print(f"  ✓ {tag} ({algo}): AR@500={m['AR@500']:5.1f}% MedAE={m['MedAE_ms']:6.0f}ms MaxAE={m['MaxAE_ms']:6.0f}ms LR={m['LR500_K5']:5.1f}% ({elapsed:.0f}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL {tag}: {e}")

        if plan_path and os.path.exists(plan_path):
            os.unlink(plan_path)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "metrics_hybrid.csv", index=False)
    print(f"\nSaved {len(df)} hybrid runs to {OUT_ROOT}/metrics_hybrid.csv")
    if len(df):
        print("\nPer-piece per-perf:")
        for _, r in df.iterrows():
            print(f"  {r['piece']}/{r['perf']}: {r['selected_algo']} → AR@500={r['AR@500']:.1f}%")


if __name__ == "__main__":
    main()
