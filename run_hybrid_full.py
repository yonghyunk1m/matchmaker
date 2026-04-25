"""Hybrid algorithm on full 159 pairs: prove "always improve" property.

Decision rule (offline, score-based):
  slow_beats(piece) = count of beats with γ_quant < 0.85 (cascade trigger)
  if slow_beats >= 50:
      → Arzt + STFT + quantitative plan via Path A (locked slow regions)
  else:
      → Dixon + CQT (paper's standard non-plan baseline)

Outputs:
  results/hybrid_full/metrics_hybrid.csv  — per-pair hybrid result
  results/hybrid_full/metrics_baseline.csv — Dixon+CQT for ALL pairs (comparison)

Goal: hybrid_AR@500 ≥ Dixon+CQT_AR@500 for every (piece, perf) → "always improve OR equal".
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

OUT_ROOT = Path("results/hybrid_full")
OUT_TRAJ = OUT_ROOT / "trajectories"
OUT_TRAJ.mkdir(parents=True, exist_ok=True)

THRESHOLD_SLOW_BEATS = 50  # cascade trigger


def build_quant_plan_with_locks(xml_path, max_beat):
    """Quantitative γ from rule lexicon + lock slow-marked regions."""
    gamma, markings = extract_rule_gamma(xml_path, max_beat)
    locked = np.zeros(max_beat, dtype=bool)
    # markings: list of (beat, kind, txt, gamma_value)
    slow_starts = [(b, txt) for (b, kind, txt, g) in markings if is_slow_marking(txt) or g < 0.85]
    all_markings = sorted(markings, key=lambda x: x[0])
    for slow_b, _ in slow_starts:
        next_b = max_beat
        for (b2, kind2, txt2, g2) in all_markings:
            if b2 > slow_b and not is_slow_marking(txt2) and g2 >= 0.85:
                next_b = int(b2)
                break
        sb = int(slow_b)
        if sb < max_beat:
            locked[sb:min(next_b, max_beat)] = True
    return gamma, locked


def select_algorithm(xml_path, max_beat):
    gamma, locked = build_quant_plan_with_locks(xml_path, max_beat)
    slow_beats = int(np.sum(gamma < 0.85))
    if slow_beats >= THRESHOLD_SLOW_BEATS:
        return 'arzt_quant_plan', gamma, locked, slow_beats
    return 'dixon_cqt', None, None, slow_beats


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
        print(f"\n{pid}: slow_beats={slow_beats} → {algo}", flush=True)

        plan_path = None
        if algo == 'arzt_quant_plan':
            dyn = np.ones(max_beat, dtype=np.float32) * 64.0
            plan_path = tempfile.mktemp(suffix=".pkl")
            with open(plan_path, "wb") as f:
                pickle.dump((gamma, dyn, locked), f)

        for perf in piece["performers"]:
            wav = os.path.join(piece["asap_dir"], f"{perf}.wav")
            if not os.path.exists(wav):
                continue
            tag_h = f"{pid}_{perf}_hybrid"
            tag_b = f"{pid}_{perf}_dixon_cqt"

            # --- Hybrid run ---
            t0 = time.time()
            try:
                if algo == 'arzt_quant_plan':
                    res = run_single(piece, perf, "xml", algorithm="arzt",
                                     plan_path_override=plan_path,
                                     per_beat_csv=str(OUT_TRAJ / f"{tag_h}.csv"))
                    df_h = pd.read_csv(OUT_TRAJ / f"{tag_h}.csv")
                    err_h = df_h["err_ms"].values
                else:
                    traj = run_dixon(piece, perf, plan_gamma=None)
                    if traj is None:
                        print(f"  SKIP {tag_h}: no audio")
                        continue
                    df_h = pd.DataFrame({k: traj[k] for k in
                                         ["beat_idx", "gt_beat", "gt_time_s", "pred_time_s", "err_ms", "ok"]})
                    df_h.to_csv(OUT_TRAJ / f"{tag_h}.csv", index=False)
                    err_h = traj["err_ms"]
                m_h = metrics_from_err(err_h)
                rows_hybrid.append({
                    "piece": pid, "perf": perf, "selected_algo": algo,
                    "slow_beats": slow_beats, "elapsed_s": round(time.time()-t0, 1), **m_h})
                print(f"  ✓ H {tag_h} ({algo}): AR@500={m_h['AR@500']:5.1f}% MedAE={m_h['MedAE_ms']:6.0f}ms", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL H {tag_h}: {e}", flush=True)
                continue

            # --- Baseline (Dixon+CQT for fairness) ---
            tagged_baseline = OUT_TRAJ / f"{tag_b}.csv"
            t0 = time.time()
            try:
                traj = run_dixon(piece, perf, plan_gamma=None)
                if traj is None:
                    continue
                df_b = pd.DataFrame({k: traj[k] for k in
                                     ["beat_idx", "gt_beat", "gt_time_s", "pred_time_s", "err_ms", "ok"]})
                df_b.to_csv(tagged_baseline, index=False)
                m_b = metrics_from_err(traj["err_ms"])
                rows_baseline.append({
                    "piece": pid, "perf": perf, "slow_beats": slow_beats,
                    "elapsed_s": round(time.time()-t0, 1), **m_b})
                print(f"  ✓ B {tag_b}: AR@500={m_b['AR@500']:5.1f}% MedAE={m_b['MedAE_ms']:6.0f}ms", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL B {tag_b}: {e}", flush=True)

        if plan_path and os.path.exists(plan_path):
            os.unlink(plan_path)

    df_h = pd.DataFrame(rows_hybrid)
    df_b = pd.DataFrame(rows_baseline)
    df_h.to_csv(OUT_ROOT / "metrics_hybrid.csv", index=False)
    df_b.to_csv(OUT_ROOT / "metrics_baseline.csv", index=False)
    print(f"\n=== Final: {len(df_h)} hybrid, {len(df_b)} baseline ===")

    if len(df_h) and len(df_b):
        merged = df_h.merge(df_b, on=["piece", "perf"], suffixes=("_hyb", "_dix"))
        merged["delta_AR500"] = merged["AR@500_hyb"] - merged["AR@500_dix"]
        n_improve = int((merged["delta_AR500"] > 0.5).sum())
        n_neutral = int((merged["delta_AR500"].abs() <= 0.5).sum())
        n_regress = int((merged["delta_AR500"] < -0.5).sum())
        print(f"\nPaired n={len(merged)}:")
        print(f"  improve  (Δ > +0.5pp): {n_improve}")
        print(f"  neutral  (|Δ| ≤ 0.5pp): {n_neutral}")
        print(f"  REGRESS  (Δ < -0.5pp): {n_regress}")
        print(f"  mean Δ: {merged['delta_AR500'].mean():+.2f}pp, min Δ: {merged['delta_AR500'].min():+.2f}pp")
        worst = merged.nsmallest(5, "delta_AR500")[["piece", "perf", "selected_algo", "AR@500_hyb", "AR@500_dix", "delta_AR500"]]
        print(f"\nWorst 5 deltas:")
        print(worst.to_string(index=False))
        merged.to_csv(OUT_ROOT / "metrics_hybrid_vs_baseline.csv", index=False)


if __name__ == "__main__":
    main()
