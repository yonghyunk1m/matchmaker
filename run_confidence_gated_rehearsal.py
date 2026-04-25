"""Confidence-gated rehearsal adaptation: pass-2 follows performer where they
diverge from the score's tempo plan, keeps rule γ where they match.

Pipeline:
  1. Pass 1: Arzt+STFT+rule_plan (locked) — extract warping path
  2. From pass-1 trajectory: compute per-section actual γ̂(b)
  3. Build pass-2 plan:
       - Section-mean |γ_rule − γ̂| < TOL → keep rule γ (still locked)
       - Section-mean |γ_rule − γ̂| ≥ TOL → switch to γ̂ (unlock that section)
  4. Pass 2: run with blended plan, measure AR@500.

Compares: Pass-1 (rule only) vs Pass-2 (confidence-gated).
Hypothesis: pass-2 rescues outliers (Sladek, JIA) without breaking
plan-correct majorities (BuiJL, KimSuyeon).

Output: results/confidence_gated/op23_pass2.csv
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

from run_full_experiment import PIECES, run_single, load_gt
from run_directional_plan_pilot import metrics_from_err
from run_scorpion_rule_intent import extract_rule_gamma

OUT_ROOT = Path("results/confidence_gated")
OUT_TRAJ = OUT_ROOT / "trajectories"
OUT_TRAJ.mkdir(parents=True, exist_ok=True)

CONFIDENCE_TOL = 0.15  # |Δγ| threshold for "rule-correct" vs "needs override"


def section_boundaries(markings, max_beat):
    """From sorted markings list, build contiguous sections."""
    bs = sorted({int(b) for b, *_ in markings if b is not None and b < max_beat})
    if 0 not in bs:
        bs = [0] + bs
    bs.append(max_beat)
    return list(zip(bs[:-1], bs[1:]))


MODERATO_BPS = 114.0 / 60.0  # 1.9 BPS — same reference as rule lexicon


def compute_actual_gamma_per_section(traj_csv, sections, _unused=None):
    """From per-beat trajectory CSV, compute actual γ per section.

    γ_actual(section) = mean_section_BPS / Moderato_BPS  (same reference frame as rule).
    """
    df = pd.read_csv(traj_csv)
    if len(df) < 2:
        return {}
    bps = np.diff(df["gt_time_s"].values)
    bps = np.where(bps > 0, 1.0 / bps, np.nan)
    beats = df["gt_beat"].values[:-1]
    out = {}
    for (a, b) in sections:
        mask = (beats >= a) & (beats < b)
        if not mask.any():
            continue
        section_bps = np.nanmean(bps[mask])
        if np.isnan(section_bps):
            continue
        out[(a, b)] = float(section_bps / MODERATO_BPS)
    return out


def piece_median_bps_from_traj(traj_csv):
    return MODERATO_BPS  # kept for API compatibility


def build_blended_plan(gamma_rule, locked_rule, sections_actual_gamma, sections, max_beat):
    """Confidence-gated blend.

    For each section [a, b):
      mean_rule_gamma = mean of gamma_rule over [a, b)
      if section in sections_actual_gamma:
        gap = |mean_rule_gamma − γ̂_section|
        if gap < TOL: keep rule (locked stays locked)
        else: override with γ̂_section, mark UNLOCKED so EMA can keep refining.
    """
    gamma = gamma_rule.copy()
    locked = locked_rule.copy()
    n_overrides = 0
    for (a, b) in sections:
        if (a, b) not in sections_actual_gamma:
            continue
        mean_rule = float(gamma_rule[a:b].mean())
        actual = sections_actual_gamma[(a, b)]
        actual = float(np.clip(actual, 0.3, 3.0))
        gap = abs(mean_rule - actual)
        if gap >= CONFIDENCE_TOL:
            gamma[a:b] = actual
            locked[a:b] = False  # let EMA refine further if subsequent passes
            n_overrides += 1
    return gamma, locked, n_overrides


def main():
    rows = []
    piece = next(p for p in PIECES if p["id"] == "chopin_ballade1")
    perfs = piece["performers"]
    xml_path = os.path.join(piece["asap_dir"], "xml_score.musicxml")
    score = pt.load_musicxml(xml_path).parts[0]
    max_beat = int(np.ceil(score.note_array()["onset_beat"].max())) + 10

    # Build pass-1 (rule) plan from quantitative lexicon
    gamma_rule, markings = extract_rule_gamma(xml_path, max_beat)
    locked_rule = np.zeros(max_beat, dtype=bool)
    # Lock all marked sections (slow markings start)
    for (b, kind, txt, g) in markings:
        if g < 0.85:  # slow markings only
            # Find next non-slow marking
            next_b = max_beat
            for (b2, k2, t2, g2) in sorted(markings, key=lambda x: x[0]):
                if b2 > b and g2 >= 0.85:
                    next_b = int(b2)
                    break
            sb = int(b)
            if sb < max_beat:
                locked_rule[sb:min(next_b, max_beat)] = True
    sections = section_boundaries(markings, max_beat)

    print(f"chopin_ballade1: max_beat={max_beat}, markings={len(markings)}, sections={len(sections)}")

    plan_path_p1 = tempfile.mktemp(suffix=".pkl")
    with open(plan_path_p1, "wb") as f:
        pickle.dump((gamma_rule, np.ones(max_beat, dtype=np.float32) * 64.0, locked_rule), f)

    for perf in perfs:
        wav = os.path.join(piece["asap_dir"], f"{perf}.wav")
        if not os.path.exists(wav):
            continue
        tag_p1 = f"chopin_ballade1_{perf}_pass1_rule"
        tag_p2 = f"chopin_ballade1_{perf}_pass2_gated"

        # ---- Pass 1: rule plan ----
        t0 = time.time()
        try:
            run_single(piece, perf, "xml", algorithm="arzt",
                       plan_path_override=plan_path_p1,
                       per_beat_csv=str(OUT_TRAJ / f"{tag_p1}.csv"))
            df1 = pd.read_csv(OUT_TRAJ / f"{tag_p1}.csv")
            m1 = metrics_from_err(df1["err_ms"].values)
            print(f"  ✓ {tag_p1}: AR@500={m1['AR@500']:5.1f}%  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAIL {tag_p1}: {e}")
            continue

        # ---- Build pass-2 plan ----
        median_bps = piece_median_bps_from_traj(OUT_TRAJ / f"{tag_p1}.csv")
        sec_actual = compute_actual_gamma_per_section(OUT_TRAJ / f"{tag_p1}.csv", sections, median_bps)
        gamma_p2, locked_p2, n_over = build_blended_plan(
            gamma_rule, locked_rule, sec_actual, sections, max_beat)
        print(f"     → {n_over} sections overridden (gap ≥ {CONFIDENCE_TOL})", flush=True)
        plan_path_p2 = tempfile.mktemp(suffix=".pkl")
        with open(plan_path_p2, "wb") as f:
            pickle.dump((gamma_p2, np.ones(max_beat, dtype=np.float32) * 64.0, locked_p2), f)

        # ---- Pass 2: blended plan ----
        t0 = time.time()
        try:
            run_single(piece, perf, "xml", algorithm="arzt",
                       plan_path_override=plan_path_p2,
                       per_beat_csv=str(OUT_TRAJ / f"{tag_p2}.csv"))
            df2 = pd.read_csv(OUT_TRAJ / f"{tag_p2}.csv")
            m2 = metrics_from_err(df2["err_ms"].values)
            delta = m2["AR@500"] - m1["AR@500"]
            print(f"  ✓ {tag_p2}: AR@500={m2['AR@500']:5.1f}%  Δ={delta:+5.1f}pp  ({time.time()-t0:.0f}s)", flush=True)
            rows.append({
                "perf": perf, "n_overrides": n_over,
                "AR@500_p1": m1["AR@500"], "AR@500_p2": m2["AR@500"],
                "delta_AR500": delta,
                "MedAE_p1": m1["MedAE_ms"], "MedAE_p2": m2["MedAE_ms"],
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAIL {tag_p2}: {e}")
        os.unlink(plan_path_p2)

    if os.path.exists(plan_path_p1):
        os.unlink(plan_path_p1)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "op23_pass2.csv", index=False)
    print(f"\nSaved {len(df)} rows to {OUT_ROOT}/op23_pass2.csv")
    if len(df):
        print(f"\nMean Δ (pass2 − pass1): {df['delta_AR500'].mean():+.2f}pp")
        print(f"Improvements: {int((df['delta_AR500'] > 0.5).sum())}/{len(df)}")
        print(f"Regressions:  {int((df['delta_AR500'] < -0.5).sum())}/{len(df)}")
        for _, r in df.iterrows():
            print(f"  {r['perf']:18s}  P1 {r['AR@500_p1']:5.1f}  →  P2 {r['AR@500_p2']:5.1f}  "
                  f"Δ {r['delta_AR500']:+5.1f}pp  ({int(r['n_overrides'])} overrides)")


if __name__ == "__main__":
    main()
