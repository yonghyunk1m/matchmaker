"""Generate a single markdown table summarising all directional/hybrid runs.

Pulls from:
  results/directional_plan/metrics_op23_n7.csv         — Arzt directional n=7 (binary γ=0.4)
  results/directional_plan/metrics.csv                  — Arzt directional pilot (Op.~23/38, Bach)
  results/directional_plan/metrics_dixon.csv            — Dixon Path B γ=0.4
  results/directional_plan/metrics_dixon_mild.csv       — Dixon Path B γ=0.7
  results/hybrid_full/metrics_hybrid.csv                — Hybrid (full pieces, growing)
  results/hybrid_full/metrics_baseline.csv              — Dixon+CQT baseline for hybrid comparison
  results/hybrid_full/metrics_hybrid_vs_baseline.csv    — Hybrid vs baseline paired
  results/audio_cascade/agreement.csv                   — Audio detector probe

Output: RESULTS_TRACKING.md (human-readable, rebuildable)
"""
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "RESULTS_TRACKING.md"


def fmt_pct(x): return f"{x:5.1f}" if pd.notnull(x) else "  -  "
def fmt_int(x): return f"{int(x):4d}" if pd.notnull(x) else "  - "
def fmt_d(x): return f"{x:+5.1f}" if pd.notnull(x) else "  -  "


def section_header(s):
    return f"\n## {s}\n"


def table_directional_n7(p):
    if not p.exists():
        return "_(no data)_\n"
    df = pd.read_csv(p)
    base = df[df.condition == "C1_baseline"].set_index("perf")
    direc = df[df.condition == "C5_directional"].set_index("perf")
    common = sorted(base.index.intersection(direc.index))
    out = ["| Performer | Baseline AR@500 | Directional AR@500 | Δ (pp) | MedAE base→dir (s) |",
           "|-----------|----------------:|-------------------:|-------:|--------------------:|"]
    for p_ in common:
        d = direc.loc[p_]["AR@500"] - base.loc[p_]["AR@500"]
        out.append(f"| {p_} | {fmt_pct(base.loc[p_]['AR@500'])} | "
                   f"{fmt_pct(direc.loc[p_]['AR@500'])} | {fmt_d(d)} | "
                   f"{base.loc[p_]['MedAE_ms']/1000:.1f} → {direc.loc[p_]['MedAE_ms']/1000:.1f} |")
    deltas = direc.loc[common]["AR@500"].values - base.loc[common]["AR@500"].values
    out.append(f"| **Mean** | **{base.loc[common]['AR@500'].mean():5.1f}** | "
               f"**{direc.loc[common]['AR@500'].mean():5.1f}** | "
               f"**{deltas.mean():+5.1f}** | — |")
    try:
        from scipy.stats import wilcoxon
        w = wilcoxon(direc.loc[common]["AR@500"].values, base.loc[common]["AR@500"].values, alternative="greater")
        out.append(f"\n**Wilcoxon (greater) p = {w.pvalue:.4f}**, n = {len(common)}")
    except Exception:
        pass
    return "\n".join(out) + "\n"


def table_dixon_path_b(path_p4, path_p7):
    rows = []
    for label, p in [("γ=0.4", path_p4), ("γ=0.7", path_p7)]:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            rows.append({"piece": r["piece"], "perf": r["perf"],
                         "label": label, "AR@500": r["AR@500"],
                         "MedAE_s": r["MedAE_ms"] / 1000, "slow_beats": int(r.get("slow_beats", 0))})
    if not rows:
        return "_(no data)_\n"
    df = pd.DataFrame(rows)
    out = ["| Piece | Perf | γ_slow | slow_beats | AR@500 | MedAE (s) |",
           "|-------|------|-------:|-----------:|-------:|----------:|"]
    for _, r in df.iterrows():
        out.append(f"| {r['piece']} | {r['perf']} | {r['label']} | "
                   f"{r['slow_beats']:>4} | {fmt_pct(r['AR@500'])} | {r['MedAE_s']:6.1f} |")
    return "\n".join(out) + "\n"


def table_hybrid_paired(p_paired):
    if not p_paired.exists():
        return "_(in progress; merged file appears after run completes)_\n"
    df = pd.read_csv(p_paired)
    if df.empty:
        return "_(empty)_\n"
    out = ["| Piece | Perf | Selected | slow_beats | Hybrid AR@500 | Dixon+CQT AR@500 | Δ (pp) |",
           "|-------|------|----------|-----------:|--------------:|-----------------:|-------:|"]
    for _, r in df.iterrows():
        out.append(f"| {r['piece']} | {r['perf']} | {r['selected_algo']} | "
                   f"{int(r['slow_beats']):>4} | {fmt_pct(r['AR@500_hyb'])} | "
                   f"{fmt_pct(r['AR@500_dix'])} | {fmt_d(r['delta_AR500'])} |")
    delta = df["delta_AR500"]
    n_imp = int((delta > 0.5).sum())
    n_neu = int((delta.abs() <= 0.5).sum())
    n_reg = int((delta < -0.5).sum())
    out.append("")
    out.append(f"**Summary**: improve={n_imp}, neutral={n_neu}, **REGRESS={n_reg}**, "
               f"mean Δ={delta.mean():+.2f}pp, min Δ={delta.min():+.2f}pp, n={len(df)}")
    return "\n".join(out) + "\n"


def table_hybrid_running(p_h, p_b):
    """Show progress before paired-merge is written."""
    if not p_h.exists():
        return "_(no rows yet)_\n"
    dh = pd.read_csv(p_h)
    db = pd.read_csv(p_b) if p_b.exists() else pd.DataFrame()
    if dh.empty:
        return "_(empty)_\n"
    out = ["| # | Piece | Perf | Selected | Hybrid AR@500 | Hybrid MedAE (s) |",
           "|---|-------|------|----------|--------------:|------------------:|"]
    for i, (_, r) in enumerate(dh.iterrows(), 1):
        out.append(f"| {i} | {r['piece']} | {r['perf']} | {r['selected_algo']} | "
                   f"{fmt_pct(r['AR@500'])} | {r['MedAE_ms']/1000:6.1f} |")
    out.append(f"\n**Hybrid runs so far**: {len(dh)} ; **Baseline pairs collected**: {len(db)}")
    return "\n".join(out) + "\n"


def table_audio_cascade(p):
    if not p.exists():
        return "_(no data)_\n"
    df = pd.read_csv(p)
    out = ["| Piece | Perf | Score-marked slow | Audio-pred | Precision | Recall | F1 |",
           "|-------|------|------------------:|-----------:|----------:|-------:|---:|"]
    for _, r in df.iterrows():
        out.append(f"| {r['piece']} | {r['perf']} | {int(r['n_slow_score']):>4} | "
                   f"{int(r['n_pred_audio']):>4} | {r['precision']:.2f} | "
                   f"{r['recall']:.2f} | {r['f1']:.2f} |")
    return "\n".join(out) + "\n"


def table_unified_dixon(p):
    if not p.exists():
        return "_(no data)_\n"
    df = pd.read_csv(p)
    if df.empty:
        return "_(empty)_\n"
    out = ["| Piece | Perf | slow_beats | Unified AR@500 | Dixon AR@500 | Δ (pp) | MedAE U/B (s) |",
           "|-------|------|-----------:|---------------:|-------------:|-------:|---------------:|"]
    for _, r in df.iterrows():
        out.append(f"| {r['piece']} | {r['perf']} | {int(r['slow_beats']):>4} | "
                   f"{fmt_pct(r['AR500_unified'])} | {fmt_pct(r['AR500_baseline'])} | "
                   f"{fmt_d(r['delta'])} | {r['MedAE_unified']/1000:.1f} / {r['MedAE_baseline']/1000:.1f} |")
    delta = df["delta"]
    out.append("")
    out.append(f"**Summary**: improve={int((delta>0.5).sum())}, neutral={int((delta.abs()<=0.5).sum())}, "
               f"**REGRESS={int((delta<-0.5).sum())}**, mean Δ={delta.mean():+.2f}pp, "
               f"min Δ={delta.min():+.2f}pp, n={len(df)}")
    return "\n".join(out) + "\n"


def table_confidence_gated(p):
    if not p.exists():
        return "_(in progress)_\n"
    df = pd.read_csv(p)
    if df.empty:
        return "_(empty)_\n"
    out = ["| Perf | n_overrides | Pass-1 AR@500 | Pass-2 (gated) AR@500 | Δ (pp) |",
           "|------|------------:|--------------:|----------------------:|-------:|"]
    for _, r in df.iterrows():
        out.append(f"| {r['perf']} | {int(r['n_overrides']):>3} | "
                   f"{fmt_pct(r['AR@500_p1'])} | {fmt_pct(r['AR@500_p2'])} | "
                   f"{fmt_d(r['delta_AR500'])} |")
    delta = df["delta_AR500"]
    out.append("")
    out.append(f"**Summary**: improve={int((delta>0.5).sum())}, neutral={int((delta.abs()<=0.5).sum())}, "
               f"REGRESS={int((delta<-0.5).sum())}, mean Δ={delta.mean():+.2f}pp, n={len(df)}")
    return "\n".join(out) + "\n"


def main():
    parts = []
    parts.append("# Results Tracking — directional plan / hybrid / audio detector / unified")
    parts.append(f"\n_Auto-generated by build_results_table.py — do not edit by hand._\n")
    parts.append("Last updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    parts.append("\n**Reporting policy**: every (piece, performer) trial is recorded — "
                 "wins, losses, and ties — across all metrics. No averaging hides outliers.")

    parts.append(section_header("1. Directional plan, Op.~23 (Arzt+STFT, binary γ=0.4)"))
    parts.append(table_directional_n7(ROOT / "results/directional_plan/metrics_op23_n7.csv"))
    parts.append("**Conclusion**: Directional binary mask gives mean +13.28pp at p=0.0547. "
                 "Quantitative lexicon (paper's existing claim, +17.3pp p=0.031 on n=6) is stronger. "
                 "→ Lexicon's BPM numbers carry information.\n")

    parts.append(section_header("2. Dixon Path B (variable-tempo audio re-render)"))
    parts.append(table_dixon_path_b(
        ROOT / "results/directional_plan/metrics_dixon.csv",
        ROOT / "results/directional_plan/metrics_dixon_mild.csv"))
    parts.append("**Conclusion**: Path B catastrophically fails on Op.~23 cascade pieces "
                 "regardless of γ value (γ=0.4 → 0.15%, γ=0.7 → 1.4%). "
                 "Re-render introduces phase artifacts incompatible with cascade structure. "
                 "Non-cascade pieces unaffected (γ never < 1.0 there).\n")

    parts.append(section_header("3. Hybrid (Arzt+quant+plan if cascade else Dixon+CQT) — full 159 pairs"))
    parts.append(table_hybrid_paired(ROOT / "results/hybrid_full/metrics_hybrid_vs_baseline.csv"))
    parts.append("\n_(running progress)_\n")
    parts.append(table_hybrid_running(
        ROOT / "results/hybrid_full/metrics_hybrid.csv",
        ROOT / "results/hybrid_full/metrics_baseline.csv"))

    parts.append(section_header("4. Unified Dixon hybrid (single tracker: Dixon norm + plan→max_run_count cap)"))
    parts.append(table_unified_dixon(ROOT / "results/unified_dixon_hybrid/metrics.csv"))
    parts.append("\n**Mechanism**: Dixon's path-length normalization (no forward bias) + "
                 "per-beat `max_run_count = floor(5γ)` cap from quantitative plan. Single tracker, "
                 "no algorithm switching.\n")

    parts.append(section_header("5. Confidence-gated rehearsal (pass-2 overrides plan-perf mismatch sections)"))
    parts.append(table_confidence_gated(ROOT / "results/confidence_gated/op23_pass2.csv"))
    parts.append("\n**Mechanism**: extract per-section actual γ from pass-1 GT timing, override "
                 "rule γ where |γ_rule − γ_actual| > 0.15, run pass-2 with blended plan. "
                 "Initial run had normalization bug (piece-median vs Moderato 114BPM); fixed in "
                 "[run_confidence_gated_rehearsal.py](run_confidence_gated_rehearsal.py).\n")

    parts.append(section_header("6. Audio-only cascade detector probe (rule-based, no training)"))
    parts.append(table_audio_cascade(ROOT / "results/audio_cascade/agreement.csv"))
    parts.append("**Conclusion**: Audio carries cascade signal — Op.~23 precision 0.50–0.55 (≫ random "
                 "0.28); non-cascade pieces (Op.~38, Bach) have 0 precision because they have 0 ground-truth "
                 "cascade beats, with FP rate ~3–5% of total beats. Threshold (z>1.0) under-recalls (only "
                 "13% of true slow beats detected). → Future work: learned classifier closes recall gap.\n")

    parts.append(section_header("7. Decisions / status"))
    parts.append("- **Directional plan (binary)**: not integrated into paper (marginal vs paper's quant claim, p=0.0547)")
    parts.append("- **Cost-bias Dixon (PlanAwareDixon)**: catastrophic, not integrated; "
                 "covered by paper §6.1 limitations")
    parts.append("- **Hybrid (algorithm switching)**: 3/6 wins, 3/6 losses on Op.~23 — "
                 "score-marking trigger insufficient for per-perf coverage")
    parts.append("- **Unified Dixon hybrid**: testing — Dixon's robustness + plan-cap "
                 "without algorithm switching")
    parts.append("- **Confidence-gated rehearsal**: testing with normalization bug fixed")
    parts.append("- **Audio detector**: future work; rule-based gives Op.~23 precision 0.5+, "
                 "non-cascade FP 3-5%\n")
    parts.append(section_header("8. Honest reporting policy"))
    parts.append("- All 159 pair-level results to be included; no subsetting beyond paper's "
                 "documented MunA19M annotation exclusion.")
    parts.append("- Per-perf deltas in supplementary table.")
    parts.append("- All metrics (AR@100/500/1000, MedAE, MaxAE, MeanAE, LR@500_K=5).")
    parts.append("- All experimental conditions reported, including failures (Path B catastrophic, "
                 "cost-bias Dixon catastrophic, directional plan marginal).\n")

    OUT.write_text("\n".join(parts))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
