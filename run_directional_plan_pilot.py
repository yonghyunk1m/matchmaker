"""Directional Plan pilot: marking-as-direction, no audio re-render.

Concept:
  Italian markings → DIRECTIONAL hint (slow / default), not BPM value.
  γ_directional(b) = 0.4 if marking ∈ {Lento, Largo, Adagio, Grave, Meno mosso}
                   = 1.0 otherwise
  Inject via Path A (Arzt step-size cap):
    δ = ⌊5γ⌋ → δ=2 in slow, δ=5 default
  No re-rendering → no numerical noise → guarantee non-marked-slow regions = baseline.

Hypothesis: Plan helps cascade pieces (marked-slow long sections)
            without harming non-cascade pieces.

Pilot: 3 pieces × 2 perfs × 2 stages.
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

from run_full_experiment import (
    PIECES, load_gt, run_single, SAMPLE_RATE,
)


OUT_ROOT = Path("results/directional_plan")
OUT_TRAJ = OUT_ROOT / "trajectories"
OUT_TRAJ.mkdir(parents=True, exist_ok=True)


SLOW_TERMS = {
    "lento", "largo", "adagio", "grave", "meno mosso",
    "andante", "andantino", "sostenuto",
    "calmando", "tranquillo", "morendo",
}


def is_slow_marking(text):
    if not text:
        return False
    t = text.lower().strip().rstrip(".,!?;:")
    return any(slow in t for slow in SLOW_TERMS)


def build_directional_plan(xml_path, max_beat, slow_gamma=0.4):
    """Generate directional γ: slow_gamma in slow-marked sections, 1.0 else."""
    score = pt.load_musicxml(xml_path)
    part = score.parts[0]
    bm = part.beat_map

    slow_starts = []
    for d in part.iter_all(pt.score.ConstantTempoDirection):
        txt = getattr(d, 'raw_text', '') or getattr(d, 'text', '')
        if d.start is None:
            continue
        beat = float(bm(d.start.t))
        if is_slow_marking(txt):
            slow_starts.append(beat)

    # Find next non-slow tempo direction (boundary)
    all_tempo = []
    for d in part.iter_all(pt.score.ConstantTempoDirection):
        if d.start is None:
            continue
        beat = float(bm(d.start.t))
        txt = getattr(d, 'raw_text', '') or getattr(d, 'text', '')
        all_tempo.append((beat, txt))
    all_tempo.sort()

    # Build γ array: default 1.0
    gamma = np.ones(max_beat, dtype=np.float32)
    locked = np.zeros(max_beat, dtype=bool)
    for slow_b in slow_starts:
        # Find next tempo marking that is NOT slow
        next_b = max_beat
        for b, txt in all_tempo:
            if b > slow_b and not is_slow_marking(txt):
                next_b = int(b)
                break
        slow_b_i = int(slow_b)
        # Apply slow γ from slow_b to next_b
        if slow_b_i < max_beat:
            gamma[slow_b_i:min(next_b, max_beat)] = slow_gamma
            locked[slow_b_i:min(next_b, max_beat)] = True

    dyn = np.ones(max_beat, dtype=np.float32) * 64.0
    return gamma, dyn, locked, len(slow_starts)


PILOT = [
    ("chopin_ballade1", 2),    # cascade - should help
    ("chopin_ballade_2", 2),   # non-cascade - should not hurt
    ("bach_prelude_848", 2),   # easy baseline - should not hurt
]


def metrics_from_err(err_ms):
    err = err_ms[~np.isnan(err_ms)]
    if len(err) == 0:
        return {}
    n = len(err_ms)
    K = 5
    within500 = (err_ms <= 500) & (~np.isnan(err_ms))
    locked = np.zeros(n, dtype=bool)
    for i in range(K - 1, n):
        locked[i] = within500[i - K + 1: i + 1].all()
    return {
        "n_beats": int(len(err_ms)),
        "AR@100": float(np.mean(err_ms <= 100) * 100),
        "AR@500": float(np.mean(err_ms <= 500) * 100),
        "AR@1000": float(np.mean(err_ms <= 1000) * 100),
        "MedAE_ms": float(np.median(err)),
        "MaxAE_ms": float(np.max(err)),
        "MeanAE_ms": float(np.mean(err)),
        "LR500_K5": float(np.sum(locked) / n * 100),
    }


def main():
    rows = []
    for pid, n_perf in PILOT:
        try:
            piece = next(p for p in PIECES if p["id"] == pid)
        except StopIteration:
            print(f"  SKIP {pid}: not in PIECES")
            continue
        perfs = piece["performers"][:n_perf]
        xml_path = os.path.join(piece["asap_dir"], "xml_score.musicxml")
        score = pt.load_musicxml(xml_path).parts[0]
        max_beat = int(np.ceil(score.note_array()["onset_beat"].max())) + 10
        gamma, dyn, locked, n_slow_markings = build_directional_plan(
            xml_path, max_beat, slow_gamma=0.4)
        slow_count = int(np.sum(locked))
        print(f"\n{pid}: max_beat={max_beat}, slow markings={n_slow_markings}, slow beats={slow_count}", flush=True)

        plan_path = tempfile.mktemp(suffix=".pkl")
        with open(plan_path, "wb") as f:
            pickle.dump((gamma, dyn, locked), f)

        for perf in perfs:
            for cond in ["C1_baseline", "C5_directional"]:
                tag = f"{pid}_{perf}_{cond}"
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
                    # Read trajectory and compute metrics
                    df = pd.read_csv(OUT_TRAJ / f"{tag}.csv")
                    m = metrics_from_err(df["err_ms"].values)
                    row = {"piece": pid, "perf": perf, "condition": cond,
                           "elapsed_s": round(elapsed, 1), "slow_beats": slow_count}
                    row.update(m)
                    rows.append(row)
                    print(f"  ✓ {tag}: AR@500={m['AR@500']:5.1f}% MedAE={m['MedAE_ms']:6.0f}ms MaxAE={m['MaxAE_ms']:6.0f}ms LR={m['LR500_K5']:5.1f}% ({elapsed:.0f}s)", flush=True)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"  FAIL {tag}: {e}", flush=True)
        os.unlink(plan_path)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "metrics.csv", index=False)
    print(f"\nSaved {len(df)} runs to {OUT_ROOT}/metrics.csv")
    if len(df) > 0:
        print("\nPer-piece per-condition mean:")
        agg = df.groupby(["piece", "condition"])[["AR@500", "MedAE_ms", "MaxAE_ms", "LR500_K5"]].mean().round(2)
        print(agg)


if __name__ == "__main__":
    main()
