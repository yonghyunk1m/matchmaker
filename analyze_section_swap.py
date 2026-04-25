"""Section-level algorithm swap analysis.

For each (piece, perf), compute per-section AR@500 using each available
tracker (Arzt+plan_v1, Pass2_gated, Dixon+CQT). For each section, pick
the best tracker. Post-hoc merge → "section-best" trajectory.

Compares against Dixon+CQT baseline for each metric.

Section boundaries from MusicXML tempo markings.

Output: results/section_swap/per_perf.csv  -  per (piece, perf)
        results/section_swap/sections.csv   -  per section choice
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import partitura as pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_experiment import PIECES
from run_scorpion_rule_intent import extract_rule_gamma

ROOT = Path(__file__).parent
OUT_ROOT = ROOT / "results" / "section_swap"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_traj(csv_path):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "err_ms" not in df.columns or "gt_beat" not in df.columns:
        return None
    return df


def section_metrics(df, a, b):
    if df is None or df.empty:
        return None
    mask = (df["gt_beat"] >= a) & (df["gt_beat"] < b)
    sub = df[mask]
    if sub.empty:
        return None
    err = sub["err_ms"].abs()
    K = 5
    ok = (err <= 500).values
    lr = sum(1 for i in range(K-1, len(ok)) if all(ok[i-K+1:i+1])) / max(len(ok)-K+1, 1) * 100
    return {
        "n_beats": len(sub),
        "AR@500": float((err <= 500).mean() * 100),
        "MedAE_s": float(err.median()) / 1000 if len(err) else np.nan,
        "MaxAE_s": float(err.max()) / 1000 if len(err) else np.nan,
        "LR_K5": float(lr),
    }


def merge_section_choices(traj_dict, sections, choices):
    """Build a synthesized trajectory by picking each section from chosen tracker."""
    chunks = []
    for (a, b), choice in zip(sections, choices):
        df = traj_dict.get(choice)
        if df is None:
            continue
        mask = (df["gt_beat"] >= a) & (df["gt_beat"] < b)
        chunks.append(df[mask])
    if not chunks:
        return None
    return pd.concat(chunks, ignore_index=True)


def total_metrics(df):
    if df is None or df.empty:
        return None
    err = df["err_ms"].abs()
    K = 5
    ok = (err <= 500).values
    lr = sum(1 for i in range(K-1, len(ok)) if all(ok[i-K+1:i+1])) / max(len(ok)-K+1, 1) * 100
    return {
        "AR@500": float((err <= 500).mean() * 100),
        "MedAE_s": float(err.median()) / 1000,
        "MaxAE_s": float(err.max()) / 1000,
        "LR_K5": float(lr),
    }


def main():
    rows_per_perf = []
    rows_per_section = []
    PIECE_IDS = sys.argv[1:] if len(sys.argv) > 1 else None
    for piece in PIECES:
        if PIECE_IDS and piece["id"] not in PIECE_IDS:
            continue
        if not PIECE_IDS:
            # Default: all pieces with at least one Pass2 trajectory
            p2_dir = ROOT / "results/confidence_gated_v2/trajectories"
            has_p2 = any(p2_dir.glob(f"{piece['id']}_*_pass2_v2.csv"))
            if not has_p2:
                continue
        xml_path = os.path.join(piece["asap_dir"], "xml_score.musicxml")
        score = pt.load_musicxml(xml_path).parts[0]
        max_beat = int(np.ceil(score.note_array()["onset_beat"].max())) + 10
        gamma_rule, markings = extract_rule_gamma(xml_path, max_beat)
        sorted_mks = sorted([(int(b), txt, g) for (b, kind, txt, g) in markings if b < max_beat],
                            key=lambda x: x[0])
        # Sections = consecutive marking ranges
        bounds = [0] + [b for (b, _, _) in sorted_mks if b > 0] + [max_beat]
        bounds = sorted(set(bounds))
        sections = list(zip(bounds[:-1], bounds[1:]))

        for perf in piece["performers"]:
            base = f"{piece['id']}_{perf}"
            p1_csv = ROOT / "results/confidence_gated_v2/trajectories" / f"{base}_pass1_v2.csv"
            p2_csv = ROOT / "results/confidence_gated_v2/trajectories" / f"{base}_pass2_v2.csv"
            dx_csv = ROOT / "results/hybrid_full/trajectories" / f"{base}_dixon_cqt.csv"

            traj = {
                "Pass1": load_traj(p1_csv),
                "Pass2": load_traj(p2_csv),
                "Dixon+CQT": load_traj(dx_csv),
            }
            if not all(v is not None for v in traj.values()):
                continue

            # Per-section best choice
            section_choices = []
            for (a, b) in sections:
                ms = {k: section_metrics(v, a, b) for k, v in traj.items()}
                ms_valid = {k: v["AR@500"] for k, v in ms.items() if v is not None}
                if not ms_valid:
                    section_choices.append("Dixon+CQT")
                    continue
                best_k = max(ms_valid, key=ms_valid.get)
                section_choices.append(best_k)
                rows_per_section.append({
                    "piece": piece["id"], "perf": perf, "section": f"{a}-{b}",
                    "AR_Pass1": ms.get("Pass1", {}).get("AR@500"),
                    "AR_Pass2": ms.get("Pass2", {}).get("AR@500"),
                    "AR_Dixon": ms.get("Dixon+CQT", {}).get("AR@500"),
                    "best_choice": best_k, "best_AR": ms[best_k]["AR@500"],
                })

            # Merge sections → synthesized trajectory
            merged = merge_section_choices(traj, sections, section_choices)
            tm = total_metrics(merged)
            dx = total_metrics(traj["Dixon+CQT"])
            p1 = total_metrics(traj["Pass1"])
            p2 = total_metrics(traj["Pass2"])

            rows_per_perf.append({
                "piece": piece["id"], "perf": perf,
                "AR_Pass1": round(p1["AR@500"], 1),
                "AR_Pass2": round(p2["AR@500"], 1),
                "AR_Dixon": round(dx["AR@500"], 1),
                "AR_SectionBest": round(tm["AR@500"], 1),
                "Δ_vs_Dixon": round(tm["AR@500"] - dx["AR@500"], 2),
                "MedAE_section_s": round(tm["MedAE_s"], 2),
                "MedAE_dixon_s": round(dx["MedAE_s"], 2),
                "MaxAE_section_s": round(tm["MaxAE_s"], 2),
                "LR_section_%": round(tm["LR_K5"], 1),
                "LR_dixon_%": round(dx["LR_K5"], 1),
                "n_sections": len(sections),
                "Pass1_picked": sum(1 for c in section_choices if c == "Pass1"),
                "Pass2_picked": sum(1 for c in section_choices if c == "Pass2"),
                "Dixon_picked": sum(1 for c in section_choices if c == "Dixon+CQT"),
            })

    df = pd.DataFrame(rows_per_perf)
    df.to_csv(OUT_ROOT / "per_perf.csv", index=False)
    pd.DataFrame(rows_per_section).to_csv(OUT_ROOT / "sections.csv", index=False)

    if df.empty:
        print("No data.")
        return
    print("Per-perf section-best results (Op.~23):\n")
    print(df.to_string(index=False))

    delta = df["Δ_vs_Dixon"]
    print(f"\nAggregate vs Dixon+CQT:")
    print(f"  improve (Δ > +0.5): {(delta>0.5).sum()}/{len(df)}")
    print(f"  neutral: {(delta.abs()<=0.5).sum()}/{len(df)}")
    print(f"  REGRESS: {(delta<-0.5).sum()}/{len(df)}    ← should be 0 for 'always improve'")
    print(f"  mean Δ:  {delta.mean():+.2f}pp")
    print(f"  min Δ:   {delta.min():+.2f}pp")


if __name__ == "__main__":
    main()
