"""Diagnose why Mo07M fails under Arzt+quant+plan but succeeds under Dixon+CQT.

Compares per-section AR@500 + actual γ vs rule γ to find the mismatch.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import partitura as pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_experiment import PIECES, load_gt
from run_scorpion_rule_intent import extract_rule_gamma
from run_directional_plan_pilot import metrics_from_err

PERF = "Mo07M"
piece = next(p for p in PIECES if p["id"] == "chopin_ballade1")
xml_path = os.path.join(piece["asap_dir"], "xml_score.musicxml")
score = pt.load_musicxml(xml_path).parts[0]
max_beat = int(np.ceil(score.note_array()["onset_beat"].max())) + 10

gamma_rule, markings = extract_rule_gamma(xml_path, max_beat)

# Build named-section boundaries
mks = sorted([(int(b), txt, g) for (b, kind, txt, g) in markings if b < max_beat],
             key=lambda x: x[0])
sections = []
for i, (b, txt, g) in enumerate(mks):
    end = mks[i + 1][0] if i + 1 < len(mks) else max_beat
    sections.append((b, end, txt, g))

print(f"Section boundaries on Op.~23 (Mo07M):\n")
print(f"{'Range':>15s}  {'Marking':<25s}  {'γ_rule':>8s}")
for (a, b, txt, g) in sections:
    print(f"{a:>5d}–{b:<5d}  ({b-a:>3d} b)  {txt:<25s}  {g:>8.3f}")

# Load Mo07M's actual playing from gt
gt_pt, gt_bt = load_gt(piece["asap_dir"], PERF)
# Per-beat actual time intervals
diffs = np.diff(gt_pt)  # seconds per beat
beats = gt_bt[:-1]
all_bps = np.where(diffs > 0, 1.0 / diffs, np.nan)
median_bps = float(np.nanmedian(all_bps))
print(f"\nPiece-median beats-per-second: {median_bps:.3f}")

# For each section, compute Mo07M's actual γ vs rule γ
print(f"\n{'Range':>15s}  {'Marking':<25s}  {'γ_rule':>8s}  {'γ_actual':>10s}  {'gap':>6s}")
for (a, b, txt, g_rule) in sections:
    mask = (beats >= a) & (beats < b)
    if not mask.any():
        continue
    section_bps = float(np.nanmean(all_bps[mask]))
    g_actual = section_bps / median_bps
    gap = abs(g_actual - g_rule)
    flag = "***" if gap > 0.15 else "   "
    print(f"{a:>5d}–{b:<5d}  ({b-a:>3d} b)  {txt:<25s}  {g_rule:>8.3f}  "
          f"{g_actual:>10.3f}  {gap:>6.3f} {flag}")

# Compare per-section AR@500 between hybrid and Dixon+CQT
print(f"\n{'='*80}\nPer-section AR@500: Hybrid vs Dixon+CQT")
print(f"{'Range':>15s}  {'Marking':<25s}  {'Hyb AR':>8s}  {'Dix AR':>8s}  {'Δ':>6s}")
df_h = pd.read_csv("results/hybrid_full/trajectories/chopin_ballade1_Mo07M_hybrid.csv")
df_d = pd.read_csv("results/hybrid_full/trajectories/chopin_ballade1_Mo07M_dixon_cqt.csv")

for (a, b, txt, g_rule) in sections:
    mh = df_h[(df_h["gt_beat"] >= a) & (df_h["gt_beat"] < b)]
    md = df_d[(df_d["gt_beat"] >= a) & (df_d["gt_beat"] < b)]
    if len(mh) == 0 or len(md) == 0:
        continue
    ar_h = (mh["err_ms"].abs() <= 500).mean() * 100
    ar_d = (md["err_ms"].abs() <= 500).mean() * 100
    delta = ar_h - ar_d
    flag = "***" if delta < -10 else "   "
    print(f"{a:>5d}–{b:<5d}  ({b-a:>3d} b)  {txt:<25s}  {ar_h:>8.1f}  {ar_d:>8.1f}  "
          f"{delta:>+6.1f} {flag}")
