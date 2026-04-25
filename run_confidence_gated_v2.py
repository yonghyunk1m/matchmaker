"""Confidence-gated rehearsal v2: bug-fixed normalization (Moderato 114BPM
matches rule lexicon reference) + cached pipeline (score audio + features
cached across pass-1/pass-2; ~30% wall-clock saving).

Pipeline:
  1. Pass 1: Arzt+STFT+rule_plan (locked) — cached score render/features
  2. Extract per-section actual γ from pass-1 traj (Moderato-relative)
  3. Override sections where |γ_rule - γ_actual| > TOL  (re-uses cached features)
  4. Pass 2 with blended plan
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

OUT_ROOT = Path("results/confidence_gated_v2")
OUT_TRAJ = OUT_ROOT / "trajectories"
OUT_TRAJ.mkdir(parents=True, exist_ok=True)

CONFIDENCE_TOL = 0.20  # slightly more conservative than v1's 0.15
MODERATO_BPS = 114.0 / 60.0  # 1.9 BPS — matches rule lexicon reference


def section_boundaries(markings, max_beat):
    bs = sorted({int(b) for (b, kind, txt, g) in markings if b is not None and b < max_beat})
    if 0 not in bs:
        bs = [0] + bs
    bs.append(max_beat)
    return list(zip(bs[:-1], bs[1:]))


def actual_gamma_per_section(traj_csv, sections):
    """Moderato-relative γ per section (matches rule lexicon)."""
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


def build_blended_plan(gamma_rule, locked_rule, sec_actual, sections, max_beat, tol=CONFIDENCE_TOL):
    gamma = gamma_rule.copy()
    locked = locked_rule.copy()
    n_overrides = 0
    overrides = []
    for (a, b) in sections:
        if (a, b) not in sec_actual:
            continue
        mean_rule = float(gamma_rule[a:b].mean())
        actual = float(np.clip(sec_actual[(a, b)], 0.3, 3.0))
        gap = abs(mean_rule - actual)
        if gap >= tol:
            gamma[a:b] = actual
            locked[a:b] = True  # keep locked in pass-2 (don't EMA further within the section)
            n_overrides += 1
            overrides.append((a, b, mean_rule, actual, gap))
    return gamma, locked, n_overrides, overrides


def main():
    rows = []
    piece = next(p for p in PIECES if p["id"] == "chopin_ballade1")
    perfs = piece["performers"]
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
    print(f"chopin_ballade1: max_beat={max_beat}, sections={len(sections)}, "
          f"locked beats={int(locked_rule.sum())}")

    plan_p1 = tempfile.mktemp(suffix=".pkl")
    with open(plan_p1, "wb") as f:
        pickle.dump((gamma_rule, np.ones(max_beat, dtype=np.float32) * 64.0, locked_rule), f)

    for perf in perfs:
        wav = os.path.join(piece["asap_dir"], f"{perf}.wav")
        if not os.path.exists(wav):
            continue
        tag1 = f"chopin_ballade1_{perf}_pass1_v2"
        tag2 = f"chopin_ballade1_{perf}_pass2_v2"

        # ---- Pass 1 ----
        t0 = time.time()
        try:
            run_single(piece, perf, "xml", algorithm="arzt",
                       plan_path_override=plan_p1,
                       per_beat_csv=str(OUT_TRAJ / f"{tag1}.csv"))
            df1 = pd.read_csv(OUT_TRAJ / f"{tag1}.csv")
            m1 = metrics_from_err(df1["err_ms"].values)
            el1 = time.time() - t0
            print(f"  ✓ {tag1}: AR@500={m1['AR@500']:5.1f}%  ({el1:.0f}s)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAIL {tag1}: {e}")
            continue

        # ---- Build pass-2 plan ----
        sec_actual = actual_gamma_per_section(OUT_TRAJ / f"{tag1}.csv", sections)
        gamma2, locked2, n_over, overrides = build_blended_plan(
            gamma_rule, locked_rule, sec_actual, sections, max_beat)
        print(f"     → {n_over} overrides (TOL={CONFIDENCE_TOL}, Moderato-relative)", flush=True)
        for (a, b, mr, ac, gap) in overrides[:5]:
            print(f"       [{a:>5}–{b:<5}] rule {mr:.3f} → actual {ac:.3f} (gap {gap:.3f})", flush=True)
        plan_p2 = tempfile.mktemp(suffix=".pkl")
        with open(plan_p2, "wb") as f:
            pickle.dump((gamma2, np.ones(max_beat, dtype=np.float32) * 64.0, locked2), f)

        # ---- Pass 2 ----
        t0 = time.time()
        try:
            run_single(piece, perf, "xml", algorithm="arzt",
                       plan_path_override=plan_p2,
                       per_beat_csv=str(OUT_TRAJ / f"{tag2}.csv"))
            df2 = pd.read_csv(OUT_TRAJ / f"{tag2}.csv")
            m2 = metrics_from_err(df2["err_ms"].values)
            d = m2["AR@500"] - m1["AR@500"]
            el2 = time.time() - t0
            print(f"  ✓ {tag2}: AR@500={m2['AR@500']:5.1f}%  Δ={d:+5.1f}pp  ({el2:.0f}s)", flush=True)
            rows.append({"perf": perf, "n_overrides": n_over,
                         "AR@500_p1": m1["AR@500"], "AR@500_p2": m2["AR@500"],
                         "delta_AR500": d,
                         "MedAE_p1": m1["MedAE_ms"], "MedAE_p2": m2["MedAE_ms"]})
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAIL {tag2}: {e}")
        os.unlink(plan_p2)

    if os.path.exists(plan_p1):
        os.unlink(plan_p1)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "op23_pass2.csv", index=False)
    print(f"\nSaved {len(df)} rows")
    if len(df):
        d = df["delta_AR500"]
        print(f"\nMean Δ: {d.mean():+.2f}pp; "
              f"improve={int((d>0.5).sum())}/{len(df)}, regress={int((d<-0.5).sum())}/{len(df)}")
        for _, r in df.iterrows():
            print(f"  {r['perf']:18s}  P1 {r['AR@500_p1']:5.1f}  P2 {r['AR@500_p2']:5.1f}  "
                  f"Δ {r['delta_AR500']:+5.1f}pp ({int(r['n_overrides'])} ov)")


if __name__ == "__main__":
    main()
