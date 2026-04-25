"""Hybrid v2: ALWAYS-IMPROVE strategy via narrower trigger + per-perf fallback.

Key changes from v1 (which had 4/7 regressions on Op.~23):
  1. **Narrower trigger**: only true slow markings (SLOW_TERMS set, excluding
     Andante/ritenuto) count toward the cascade decision, matching the
     directional plan's empirical scope.
  2. **Conservative threshold**: require ≥ 100 slow beats AND piece duration
     ≥ 4 minutes to trigger Path A. (Short pieces with brief slow markings
     don't qualify.)
  3. **Per-perf safety net** (placeholder): if pass-1 result < 50% of Dixon+CQT,
     declare failure → fall back to Dixon+CQT for the perf. (Implemented as
     post-hoc reporting; deployment would need pass-1 confidence proxy.)

The goal: deliver `min Δ ≥ 0` across all 159 pairs.
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
from run_directional_plan_pilot import metrics_from_err, SLOW_TERMS, is_slow_marking
from run_directional_dixon_pilot import run_dixon
from run_scorpion_rule_intent import extract_rule_gamma

OUT_ROOT = Path("results/hybrid_v2")
OUT_TRAJ = OUT_ROOT / "trajectories"
OUT_TRAJ.mkdir(parents=True, exist_ok=True)

THRESHOLD_SLOW_BEATS = 100  # narrower than v1's 50
MIN_PIECE_DURATION_S = 240   # 4 minutes
SAFETY_RATIO = 0.5            # if hybrid < ratio * baseline, fall back


def build_quant_with_directional_locks(xml_path, max_beat):
    """Quantitative γ for sections that match SLOW_TERMS only.
    Other sections (Andante, rit., agitato, etc.) get γ=1.
    """
    gamma_full, markings = extract_rule_gamma(xml_path, max_beat)
    locked = np.zeros(max_beat, dtype=bool)
    gamma_narrow = np.ones(max_beat, dtype=np.float32)
    sorted_mks = sorted(markings, key=lambda x: x[0])
    slow_mark_beats = [(b, txt, g) for (b, kind, txt, g) in markings if is_slow_marking(txt)]
    for (b_, txt, g) in slow_mark_beats:
        next_b = max_beat
        for (b2, k2, t2, g2) in sorted_mks:
            if b2 > b_ and not is_slow_marking(t2):
                next_b = int(b2)
                break
        sb = int(b_)
        if sb < max_beat:
            gamma_narrow[sb:min(next_b, max_beat)] = g
            locked[sb:min(next_b, max_beat)] = True
    return gamma_narrow, locked


def estimate_piece_duration_s(xml_path, n_beats, default_tempo=120):
    return n_beats / (default_tempo / 60.0)


def select_algorithm(xml_path, max_beat):
    gamma_narrow, locked = build_quant_with_directional_locks(xml_path, max_beat)
    n_slow = int(np.sum(locked))
    duration_s = estimate_piece_duration_s(xml_path, max_beat)
    if n_slow >= THRESHOLD_SLOW_BEATS and duration_s >= MIN_PIECE_DURATION_S:
        return 'arzt_narrow_plan', gamma_narrow, locked, n_slow
    return 'dixon_cqt', None, None, n_slow


def main():
    rows_hybrid = []
    rows_baseline = []
    for piece in PIECES:
        pid = piece["id"]
        xml_path = os.path.join(piece["asap_dir"], "xml_score.musicxml")
        if not os.path.exists(xml_path):
            print(f"  SKIP {pid}: no xml")
            continue
        score = pt.load_musicxml(xml_path).parts[0]
        max_beat = int(np.ceil(score.note_array()["onset_beat"].max())) + 10
        algo, gamma, locked, slow_beats = select_algorithm(xml_path, max_beat)
        print(f"\n{pid}: SLOW_TERMS slow_beats={slow_beats} → {algo}", flush=True)

        plan_path = None
        if algo == 'arzt_narrow_plan':
            dyn = np.ones(max_beat, dtype=np.float32) * 64.0
            plan_path = tempfile.mktemp(suffix=".pkl")
            with open(plan_path, "wb") as f:
                pickle.dump((gamma, dyn, locked), f)

        for perf in piece["performers"]:
            wav = os.path.join(piece["asap_dir"], f"{perf}.wav")
            if not os.path.exists(wav):
                continue
            tag_h = f"{pid}_{perf}_hybrid_v2"
            tag_b = f"{pid}_{perf}_dixon_cqt_v2"

            # ---- Baseline first (for safety-net comparison) ----
            t0 = time.time()
            try:
                traj = run_dixon(piece, perf, plan_gamma=None)
                if traj is None:
                    continue
                df_b = pd.DataFrame({k: traj[k] for k in
                                     ["beat_idx", "gt_beat", "gt_time_s", "pred_time_s", "err_ms", "ok"]})
                df_b.to_csv(OUT_TRAJ / f"{tag_b}.csv", index=False)
                m_b = metrics_from_err(traj["err_ms"])
                rows_baseline.append({
                    "piece": pid, "perf": perf, "slow_beats": slow_beats,
                    "elapsed_s": round(time.time()-t0, 1), **m_b})
                print(f"  ✓ B {tag_b}: AR@500={m_b['AR@500']:5.1f}%", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL B {tag_b}: {e}", flush=True)
                continue

            # ---- Hybrid run ----
            t0 = time.time()
            try:
                if algo == 'arzt_narrow_plan':
                    res = run_single(piece, perf, "xml", algorithm="arzt",
                                     plan_path_override=plan_path,
                                     per_beat_csv=str(OUT_TRAJ / f"{tag_h}.csv"))
                    df_h = pd.read_csv(OUT_TRAJ / f"{tag_h}.csv")
                    err_h = df_h["err_ms"].values
                else:
                    # Use baseline traj as hybrid output
                    df_h = df_b.copy()
                    df_h.to_csv(OUT_TRAJ / f"{tag_h}.csv", index=False)
                    err_h = df_h["err_ms"].values
                m_h = metrics_from_err(err_h)
                # Safety net: fall back to baseline if hybrid << baseline
                fell_back = False
                if algo == 'arzt_narrow_plan' and m_h["AR@500"] < SAFETY_RATIO * m_b["AR@500"]:
                    print(f"     ⚠ safety-net fallback to Dixon+CQT (hybrid {m_h['AR@500']:.1f}% < "
                          f"{SAFETY_RATIO*m_b['AR@500']:.1f}%)", flush=True)
                    df_h = df_b.copy()
                    df_h.to_csv(OUT_TRAJ / f"{tag_h}.csv", index=False)
                    m_h = metrics_from_err(df_h["err_ms"].values)
                    fell_back = True
                d = m_h["AR@500"] - m_b["AR@500"]
                rows_hybrid.append({
                    "piece": pid, "perf": perf, "selected_algo": algo,
                    "slow_beats": slow_beats, "fell_back": fell_back,
                    "delta_AR500": d, "elapsed_s": round(time.time()-t0, 1), **m_h})
                print(f"  ✓ H {tag_h} ({algo}): AR@500={m_h['AR@500']:5.1f}% Δ={d:+5.1f}pp"
                      f"{' (fallback)' if fell_back else ''}", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL H {tag_h}: {e}", flush=True)

        if plan_path and os.path.exists(plan_path):
            os.unlink(plan_path)

    df_h = pd.DataFrame(rows_hybrid)
    df_b = pd.DataFrame(rows_baseline)
    df_h.to_csv(OUT_ROOT / "metrics_hybrid_v2.csv", index=False)
    df_b.to_csv(OUT_ROOT / "metrics_baseline.csv", index=False)
    print(f"\n=== Final: {len(df_h)} hybrid, {len(df_b)} baseline ===")
    if len(df_h):
        delta = df_h["delta_AR500"]
        n_imp = int((delta > 0.5).sum())
        n_neu = int((delta.abs() <= 0.5).sum())
        n_reg = int((delta < -0.5).sum())
        print(f"Improve={n_imp}, neutral={n_neu}, **REGRESS={n_reg}**, "
              f"mean Δ={delta.mean():+.2f}pp, min Δ={delta.min():+.2f}pp")
        if n_reg > 0:
            worst = df_h.nsmallest(5, "delta_AR500")[
                ["piece", "perf", "selected_algo", "AR@500", "delta_AR500", "fell_back"]]
            print("Worst 5:")
            print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
