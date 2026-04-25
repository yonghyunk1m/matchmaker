"""Extend directional Arzt pilot to all 7 Op.~23 performers (n=7)."""
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
from run_directional_plan_pilot import build_directional_plan, metrics_from_err

OUT_ROOT = Path("results/directional_plan")
OUT_TRAJ = OUT_ROOT / "trajectories"


def main():
    rows = []
    piece = next(p for p in PIECES if p["id"] == "chopin_ballade1")
    perfs = piece["performers"]  # all 7
    xml_path = os.path.join(piece["asap_dir"], "xml_score.musicxml")
    score = pt.load_musicxml(xml_path).parts[0]
    max_beat = int(np.ceil(score.note_array()["onset_beat"].max())) + 10
    gamma, dyn, locked, n_slow_markings = build_directional_plan(
        xml_path, max_beat, slow_gamma=0.4)
    slow_count = int(np.sum(locked))
    print(f"chopin_ballade1: max_beat={max_beat}, slow markings={n_slow_markings}, slow beats={slow_count}", flush=True)

    plan_path = tempfile.mktemp(suffix=".pkl")
    with open(plan_path, "wb") as f:
        pickle.dump((gamma, dyn, locked), f)

    for perf in perfs:
        for cond in ["C1_baseline", "C5_directional"]:
            tag = f"chopin_ballade1_{perf}_{cond}"
            existing = OUT_TRAJ / f"{tag}.csv"
            if existing.exists():
                df = pd.read_csv(existing)
                m = metrics_from_err(df["err_ms"].values)
                row = {"piece": "chopin_ballade1", "perf": perf, "condition": cond,
                       "elapsed_s": 0.0, "slow_beats": slow_count}
                row.update(m)
                rows.append(row)
                print(f"  ◇ {tag} (cached): AR@500={m['AR@500']:5.1f}%", flush=True)
                continue
            t0 = time.time()
            try:
                if cond == "C1_baseline":
                    res = run_single(piece, perf, "xml", algorithm="arzt",
                                     per_beat_csv=str(OUT_TRAJ / f"{tag}.csv"))
                else:
                    res = run_single(piece, perf, "xml", algorithm="arzt",
                                     plan_path_override=plan_path,
                                     per_beat_csv=str(OUT_TRAJ / f"{tag}.csv"))
                elapsed = time.time() - t0
                df = pd.read_csv(OUT_TRAJ / f"{tag}.csv")
                m = metrics_from_err(df["err_ms"].values)
                row = {"piece": "chopin_ballade1", "perf": perf, "condition": cond,
                       "elapsed_s": round(elapsed, 1), "slow_beats": slow_count}
                row.update(m)
                rows.append(row)
                print(f"  ✓ {tag}: AR@500={m['AR@500']:5.1f}% MedAE={m['MedAE_ms']:6.0f}ms ({elapsed:.0f}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL {tag}: {e}", flush=True)
    os.unlink(plan_path)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "metrics_op23_n7.csv", index=False)
    print(f"\nSaved {len(df)} runs to {OUT_ROOT}/metrics_op23_n7.csv")

    # Paired comparison
    base = df[df.condition == "C1_baseline"].set_index("perf")["AR@500"]
    direc = df[df.condition == "C5_directional"].set_index("perf")["AR@500"]
    common = base.index.intersection(direc.index)
    deltas = direc.loc[common] - base.loc[common]
    print(f"\nPaired n={len(common)}: baseline mean={base.loc[common].mean():.2f}%, directional mean={direc.loc[common].mean():.2f}%, delta mean={deltas.mean():+.2f}pp")
    print("Per-perf:")
    for p in common:
        print(f"  {p:18s}  base {base[p]:5.2f}%  →  dir {direc[p]:5.2f}%  Δ {direc[p]-base[p]:+6.2f}pp")
    try:
        from scipy.stats import wilcoxon
        w = wilcoxon(direc.loc[common], base.loc[common], alternative="greater")
        print(f"Wilcoxon (greater): p={w.pvalue:.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
