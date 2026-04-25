"""Full trajectory experiment: 7 pieces × 3 perfs × 4 stages.

Stages:
  C1: Baseline (Dixon+CQT, no plan, no retempo)
  C2: Global retempo (Pass-1 ρ → reference re-rendered at ρ·T_0)
  C3: Plan via Path B (rule γ → variable-tempo reference)
  C4: Plan + Global retempo (rule γ + Pass-1 ρ refinement)

Saves per-beat trajectory + aggregate metrics.

Output:
  results/comprehensive_trajectory/
    trajectories/{piece}_{perf}_{stage}.csv
    metrics_aggregate.csv (appended)
    config_full.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_experiment import (
    PIECES, load_gt, SAMPLE_RATE, _get_xml_tempo,
    generate_score_audio_variable,
    _build_score_times_for_beats_variable,
    _build_score_times_for_beats,
)
from run_scorpion_rule_intent import extract_rule_gamma
from matchmaker import Matchmaker
from matchmaker.dp import OnlineTimeWarpingDixon
from matchmaker.utils.eval import transfer_from_score_to_predicted_perf
from matchmaker.features.audio import FRAME_RATE, ChromagramProcessor
import matchmaker.matchmaker as _mm_mod
import partitura as pt
import librosa


OUT_ROOT = Path("results/comprehensive_trajectory")
OUT_TRAJ = OUT_ROOT / "trajectories"
OUT_GAMMA = OUT_ROOT / "gamma_profiles"
OUT_TRAJ.mkdir(parents=True, exist_ok=True)
OUT_GAMMA.mkdir(parents=True, exist_ok=True)


def gamma_to_tempo_map(gamma, base_tempo):
    """Convert per-beat γ array to (beat, bpm) tempo_map for variable-tempo synth.
    Compresses runs of constant γ into change-points only.
    """
    tempo_map = [(0.0, base_tempo * float(gamma[0]))]
    for i in range(1, len(gamma)):
        if gamma[i] != gamma[i - 1]:
            tempo_map.append((float(i), base_tempo * float(gamma[i])))
    return tempo_map


def cqt_processor():
    """Return CQT chroma processor (same as run_cqt_native)."""
    from run_cqt_native import CQTChromagramProcessor, StreamCQTChromagramProcessor
    return CQTChromagramProcessor(sample_rate=SAMPLE_RATE), StreamCQTChromagramProcessor


def build_mm(xml_path, wav_path, tempo, score_audio_override=None, use_cqt=True):
    """Build Matchmaker with optional pre-rendered score audio override."""
    _mm_mod.DEFAULT_TEMPO = tempo
    mm = Matchmaker(score_file=xml_path, performance_file=wav_path,
                    input_type="audio", wait=False)
    na = mm.score_part.note_array()
    first_beat = na["onset_beat"].min()
    bt = mm.score_part.time_sigs[0].beat_type if mm.score_part.time_sigs else 4
    trim = max(first_beat, 0) * (60.0 / tempo)
    ts = int(trim * SAMPLE_RATE)
    if score_audio_override is not None:
        mm.score_audio = score_audio_override
    if ts > 0 and score_audio_override is None:
        mm.score_audio = mm.score_audio[ts:]
    if use_cqt:
        proc, StreamCQT = cqt_processor()
        mm.processor = proc
        mm.reference_features = mm.processor(mm.score_audio)
        perf_y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        mm.stream.processor = StreamCQT(perf_y, sample_rate=SAMPLE_RATE)
    else:
        mm.reference_features = mm.processor(mm.score_audio)
    mm.score_follower = OnlineTimeWarpingDixon(
        reference_features=mm.reference_features, queue=mm.stream.queue)
    return mm, na, first_beat, bt, trim


def compute_pred(mm, na, first_beat, bt, trim, gt_pt, gt_bt, tempo, score_audio_dur=None):
    """After running mm, compute predicted times + per-beat err."""
    wp = np.array(mm.score_follower.warping_path)
    sp_bmin = max(na["onset_beat"].min(), 0.0)
    sp_bmax = na["onset_beat"].max()
    lut_b = np.linspace(sp_bmin, sp_bmax, 2000)
    lut_t = _build_score_times_for_beats(mm.score_part, lut_b, tempo)
    gt_beats_scaled = gt_bt * (bt / 4)
    score_annots = np.interp(gt_beats_scaled, lut_b, lut_t) - trim
    if score_audio_dur is None:
        score_audio_dur = len(mm.score_audio) / SAMPLE_RATE
    score_annots = np.clip(score_annots, 0, score_audio_dur)
    pred = transfer_from_score_to_predicted_perf(wp, score_annots, FRAME_RATE)
    err_ms = np.where(np.isnan(pred), np.nan, np.abs(gt_pt - pred) * 1000.0)
    return pred, err_ms


def run_stage(piece, performer, stage):
    """Run one stage. Returns dict of per-beat traj or None on failure."""
    piece_dir = piece["asap_dir"]
    xml_path = os.path.join(piece_dir, "xml_score.musicxml")
    wav_path = os.path.join(piece_dir, f"{performer}.wav")
    if not os.path.exists(wav_path):
        return None

    initial_tempo = _get_xml_tempo(xml_path)
    use_resynth_global = stage in ("C2", "C4")
    use_plan = stage in ("C3", "C4")

    # Determine score audio (default None = MM will synth at default tempo)
    score_audio_override = None
    tempo_for_eval = initial_tempo

    # If plan: pre-render variable-tempo score audio
    if use_plan:
        score_part = pt.load_musicxml(xml_path).parts[0]
        max_beat = int(np.ceil(score_part.note_array()["onset_beat"].max())) + 2
        gamma, _ = extract_rule_gamma(xml_path, max_beat)
        # Save γ profile (once per piece)
        gamma_csv = OUT_GAMMA / f"{piece['id']}_rule.csv"
        if not gamma_csv.exists():
            pd.DataFrame({"beat": np.arange(len(gamma)), "gamma": gamma}).to_csv(gamma_csv, index=False)
        # Build tempo_map and synthesize variable-tempo audio
        tempo_map = gamma_to_tempo_map(gamma, initial_tempo)
        score_audio_override = generate_score_audio_variable(
            score_part, tempo_map, SAMPLE_RATE)
        # For eval, use mean tempo as effective initial
        tempo_for_eval = initial_tempo  # γ already baked into audio

    # If global retempo: do Pass-1 to estimate ρ
    if use_resynth_global:
        mm0, _, _, _, _ = build_mm(xml_path, wav_path, initial_tempo,
                                    score_audio_override=score_audio_override, use_cqt=True)
        for _ in mm0.run(verbose=False):
            pass
        wp0 = np.array(mm0.score_follower.warping_path)
        if wp0[1][-1] > 0:
            rho = float(wp0[0][-1] / wp0[1][-1])
            tempo_for_eval = initial_tempo * rho
            # Re-render score audio at rho * initial_tempo
            if use_plan:
                # Path B + retempo: scale tempo_map by rho
                tempo_map_scaled = [(b, bpm * rho) for b, bpm in tempo_map]
                score_audio_override = generate_score_audio_variable(
                    score_part, tempo_map_scaled, SAMPLE_RATE)
            else:
                score_audio_override = None  # MM will synth at tempo_for_eval

    # Pass 2 (or only pass): build mm and run
    mm, na, first_beat, bt, trim = build_mm(
        xml_path, wav_path, tempo_for_eval,
        score_audio_override=score_audio_override, use_cqt=True)
    for _ in mm.run(verbose=False):
        pass

    gt_pt, gt_bt = load_gt(piece_dir, performer)
    pred, err_ms = compute_pred(mm, na, first_beat, bt, trim, gt_pt, gt_bt, tempo_for_eval)

    return {
        "beat_idx": np.arange(len(gt_pt)),
        "gt_beat": gt_bt,
        "gt_time_s": gt_pt,
        "pred_time_s": pred,
        "err_ms": err_ms,
        "ok": (~np.isnan(pred)).astype(int),
    }


def metrics_from_err(err_ms):
    err = err_ms[~np.isnan(err_ms)]
    if len(err) == 0:
        return {}
    n = len(err_ms)
    within500 = (err_ms <= 500) & (~np.isnan(err_ms))
    K = 5
    locked = np.zeros(n, dtype=bool)
    for i in range(K - 1, n):
        locked[i] = within500[i - K + 1: i + 1].all()
    return {
        "n_beats": int(len(err_ms)),
        "n_valid": int(len(err)),
        "AR@100": float(np.mean(err_ms <= 100) * 100),
        "AR@250": float(np.mean(err_ms <= 250) * 100),
        "AR@500": float(np.mean(err_ms <= 500) * 100),
        "AR@1000": float(np.mean(err_ms <= 1000) * 100),
        "AR@2000": float(np.mean(err_ms <= 2000) * 100),
        "MedAE_ms": float(np.median(err)),
        "MaxAE_ms": float(np.max(err)),
        "MeanAE_ms": float(np.mean(err)),
        "StdAE_ms": float(np.std(err)),
        "LR500_K5": float(np.sum(locked) / n * 100),
    }


# Full grid
PIECES_FULL = [
    "chopin_ballade1",   # silent-gap cascade (focus)
    "chopin_ballade_2",
    "chopin_ballade_3",  # may not exist
    "chopin_ballade4",   # may not exist
    "chopin_scherzo_2",
    "schubert_imp_90_3",
    "liszt_campanella",
    "bach_prelude_848",
]
PERF_PER_PIECE = 3
STAGES = ["C1", "C2", "C3", "C4"]


def main():
    # Validate pieces exist in PIECES
    valid_pieces = []
    for pid in PIECES_FULL:
        try:
            piece = next(p for p in PIECES if p["id"] == pid)
            valid_pieces.append(piece)
        except StopIteration:
            print(f"  SKIP: piece {pid} not in PIECES list")
    config = {
        "pieces": [p["id"] for p in valid_pieces],
        "perf_per_piece": PERF_PER_PIECE,
        "stages": STAGES,
        "stages_def": {
            "C1": "Baseline (Dixon+CQT, no plan, no retempo)",
            "C2": "Global retempo (Pass-1 ρ → reference re-render)",
            "C3": "Plan via Path B (rule γ → variable-tempo reference)",
            "C4": "Plan + Global retempo (rule γ + Pass-1 ρ refinement)",
        },
    }
    with open(OUT_ROOT / "config_full.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load existing aggregate to avoid re-running
    agg_csv = OUT_ROOT / "metrics_aggregate.csv"
    existing = set()
    if agg_csv.exists():
        df_old = pd.read_csv(agg_csv)
        for _, r in df_old.iterrows():
            existing.add((r["piece"], r["perf"], r["condition"]))
        print(f"Resuming with {len(existing)} existing runs cached.")
    rows_new = []

    for piece in valid_pieces:
        pid = piece["id"]
        perfs = piece["performers"][:PERF_PER_PIECE]
        for perf in perfs:
            for stage in STAGES:
                # Map stage label to config key
                cond_label = {"C1": "C1_baseline", "C2": "C2_global_retempo",
                              "C3": "C3_plan_path_b", "C4": "C4_plan_plus_retempo"}[stage]
                if (pid, perf, cond_label) in existing:
                    print(f"  CACHE {pid}/{perf}/{cond_label}", flush=True)
                    continue
                tag = f"{pid}_{perf}_{cond_label}"
                t0 = time.time()
                try:
                    traj = run_stage(piece, perf, stage)
                    if traj is None:
                        print(f"  SKIP {tag}: no audio", flush=True)
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

                    m = metrics_from_err(traj["err_ms"])
                    elapsed = time.time() - t0
                    row = {"piece": pid, "perf": perf, "condition": cond_label,
                           "elapsed_s": round(elapsed, 1)}
                    row.update(m)
                    rows_new.append(row)
                    print(f"  ✓ {tag}: AR@500={m['AR@500']:5.1f}% MedAE={m['MedAE_ms']:6.0f}ms MaxAE={m['MaxAE_ms']:6.0f}ms ({elapsed:.0f}s)", flush=True)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"  FAIL {tag}: {e}", flush=True)

    # Append new rows to aggregate
    if rows_new:
        df_new = pd.DataFrame(rows_new)
        if agg_csv.exists():
            df_old = pd.read_csv(agg_csv)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(agg_csv, index=False)
        print(f"\nAppended {len(rows_new)} new runs. Aggregate now {len(df_all)} rows.")


if __name__ == "__main__":
    main()
