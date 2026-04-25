"""Validate that transcription-based GT gives the same per-section algorithm
selection as real MIDI GT. If agreement is high (>90%), section-swap is
deployable with pure audio (no MIDI keyboard needed).

Methodology:
  1. For each (piece, perf) with Pass1, Pass2, Dixon trajectories:
     - Real GT: use ASAP MIDI annotations as ground truth
     - Transcribed GT: use audio-to-MIDI transcription (Onsets&Frames)
  2. For each section, compute AR@500 against both GTs.
  3. Compare per-section algorithm choices.
  4. Report agreement rate.

If agreement > 90% → transcription-based deployment is viable.
"""
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import partitura as pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_experiment import PIECES, load_gt
from run_scorpion_rule_intent import extract_rule_gamma
from analyze_section_swap import section_metrics, total_metrics

OUT_ROOT = Path("results/transcription_validation")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
TRANSCRIPTION_CACHE = OUT_ROOT / "midi_cache"
TRANSCRIPTION_CACHE.mkdir(exist_ok=True)


def transcribe_audio(wav_path, out_midi):
    """Audio-to-MIDI transcription using piano_transcription_inference."""
    if out_midi.exists():
        return  # cached
    from piano_transcription_inference import PianoTranscription, sample_rate
    y, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    transcriptor = PianoTranscription(device="cpu", checkpoint_path=None)
    transcriptor.transcribe(y, str(out_midi))


def get_transcribed_beat_times(midi_path, n_beats):
    """Map transcribed note onsets to (approximate) beat onsets via interpolation
    against a uniform beat grid spanning the transcribed time range.
    Simple approach — assumes constant tempo.
    """
    import pretty_midi
    mid = pretty_midi.PrettyMIDI(str(midi_path))
    onsets = sorted([n.start for inst in mid.instruments for n in inst.notes])
    if len(onsets) < n_beats:
        # Pad with linear extrapolation
        return None
    # Use note onsets as proxy beat times — sample n_beats evenly
    indices = np.linspace(0, len(onsets) - 1, n_beats).astype(int)
    return np.array([onsets[i] for i in indices])


def real_gt_beat_times(piece, perf):
    gt_pt, _ = load_gt(piece["asap_dir"], perf)
    return gt_pt


def compute_per_section_ar500(traj_csv, gt_pt, sections):
    """For each section, compute AR@500 of trajectory against given GT."""
    df = pd.read_csv(traj_csv)
    if df.empty or "pred_time_s" not in df.columns:
        return {}
    pred = df["pred_time_s"].values
    gt_beats = df["gt_beat"].values
    out = {}
    for (a, b) in sections:
        mask = (gt_beats >= a) & (gt_beats < b)
        sub_pred = pred[mask]
        sub_idx = np.where(mask)[0]
        if len(sub_idx) == 0:
            continue
        # Match each pred beat to corresponding gt time (use beat index)
        # Note: gt_pt is indexed by perf-beat; pred is indexed by score-beat
        # For section-AR computation, we use the trajectory's stored err_ms
        # OR if comparing against a different GT, recompute err
        if len(sub_pred) == len(sub_idx):
            err = np.abs(np.where(np.isnan(sub_pred), 1e9, sub_pred * 1000.0
                                   - gt_pt[sub_idx[0]:sub_idx[0]+len(sub_pred)] * 1000.0))
            ar = float((err <= 500).mean() * 100)
            out[(a, b)] = ar
    return out


def main():
    rows = []
    # Use just BuiJL for initial validation
    PILOT = [("chopin_ballade1", ["BuiJL04M", "JIA06M", "KimSuyeon04M",
                                    "Mo07M", "Richardson13M", "Sladek04M", "Zhou06M"])]
    for pid, perfs in PILOT:
        piece = next(p for p in PIECES if p["id"] == pid)
        xml_path = os.path.join(piece["asap_dir"], "xml_score.musicxml")
        score = pt.load_musicxml(xml_path).parts[0]
        max_beat = int(np.ceil(score.note_array()["onset_beat"].max())) + 10
        gamma_rule, markings = extract_rule_gamma(xml_path, max_beat)
        sorted_mks = sorted([(int(b), txt, g) for (b, kind, txt, g) in markings if b < max_beat],
                            key=lambda x: x[0])
        bounds = [0] + [b for (b, _, _) in sorted_mks if b > 0] + [max_beat]
        bounds = sorted(set(bounds))
        sections = list(zip(bounds[:-1], bounds[1:]))

        for perf in perfs:
            wav = os.path.join(piece["asap_dir"], f"{perf}.wav")
            if not os.path.exists(wav):
                continue
            print(f"\n{pid}/{perf}:")

            # Transcribe (cache)
            midi_out = TRANSCRIPTION_CACHE / f"{pid}_{perf}.mid"
            t0 = time.time()
            try:
                transcribe_audio(wav, midi_out)
                print(f"  transcribed in {time.time()-t0:.0f}s")
            except Exception as e:
                print(f"  FAIL transcription: {e}")
                continue

            # Real GT and transcribed GT
            real_gt = real_gt_beat_times(piece, perf)
            transcribed_gt = get_transcribed_beat_times(midi_out, len(real_gt))
            if transcribed_gt is None:
                continue

            # Per-section selection: compare what algorithm wins under each GT
            p1_csv = Path(f"results/confidence_gated_v2/trajectories/{pid}_{perf}_pass1_v2.csv")
            p2_csv = Path(f"results/confidence_gated_v2/trajectories/{pid}_{perf}_pass2_v2.csv")
            dx_csv = Path(f"results/hybrid_full/trajectories/{pid}_{perf}_dixon_cqt.csv")
            if not all(p.exists() for p in [p1_csv, p2_csv, dx_csv]):
                continue

            # For section-best with REAL GT (already in trajectory CSVs as err_ms)
            real_choices = []
            transcribed_choices = []
            for (a, b) in sections:
                # Real GT scoring: use existing err_ms which is vs real GT
                p1_real = section_metrics(pd.read_csv(p1_csv), a, b)
                p2_real = section_metrics(pd.read_csv(p2_csv), a, b)
                dx_real = section_metrics(pd.read_csv(dx_csv), a, b)
                if not all([p1_real, p2_real, dx_real]):
                    continue
                real_scores = {"Pass1": p1_real["AR@500"],
                               "Pass2": p2_real["AR@500"],
                               "Dixon": dx_real["AR@500"]}
                real_choices.append(max(real_scores, key=real_scores.get))

                # Transcribed GT scoring: recompute err with transcribed beat times
                # Approximate: just use the beat-indexed error against transcribed beat times
                def trans_ar500(traj_csv):
                    df = pd.read_csv(traj_csv)
                    pred = df["pred_time_s"].values
                    gt_beats = df["gt_beat"].values
                    mask = (gt_beats >= a) & (gt_beats < b)
                    sub_pred = pred[mask]
                    sub_idx = np.where(mask)[0]
                    if len(sub_pred) == 0 or len(transcribed_gt) <= sub_idx.max():
                        return 0.0
                    sub_gt = transcribed_gt[sub_idx]
                    err = np.where(np.isnan(sub_pred), 1e9, np.abs(sub_pred - sub_gt) * 1000.0)
                    return float((err <= 500).mean() * 100)
                trans_scores = {"Pass1": trans_ar500(p1_csv),
                                "Pass2": trans_ar500(p2_csv),
                                "Dixon": trans_ar500(dx_csv)}
                transcribed_choices.append(max(trans_scores, key=trans_scores.get))

            if not real_choices:
                continue
            agree = sum(1 for r, t in zip(real_choices, transcribed_choices) if r == t)
            agreement_rate = agree / len(real_choices) * 100
            rows.append({
                "piece": pid, "perf": perf,
                "n_sections": len(real_choices),
                "agreement_count": agree,
                "agreement_rate": round(agreement_rate, 1),
                "real_dist": str(pd.Series(real_choices).value_counts().to_dict()),
                "transcribed_dist": str(pd.Series(transcribed_choices).value_counts().to_dict()),
            })
            print(f"  agreement: {agree}/{len(real_choices)} ({agreement_rate:.1f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "agreement.csv", index=False)
    if rows:
        mean_agree = df["agreement_rate"].mean()
        print(f"\n=== Cross-perf transcription→selection agreement ===")
        print(f"Mean agreement: {mean_agree:.1f}%")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
