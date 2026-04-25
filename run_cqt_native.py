"""
CQT Native Pipeline — Inject CQTChromagramProcessor into Matchmaker
=====================================================================
Instead of manual queue feeding (slow, timeout on long pieces),
directly replace Matchmaker's processor with CQT version.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from run_full_experiment import (
    PIECES, OUTPUT_DIR, _get_xml_tempo, load_gt,
    _build_score_times_for_beats,
)
from matchmaker import Matchmaker
from matchmaker.dp import OnlineTimeWarpingDixon
from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArzt
from matchmaker.features.audio import CQTChromagramProcessor, StreamCQTChromagramProcessor
from matchmaker.matchmaker import SAMPLE_RATE
from matchmaker.utils.eval import transfer_from_score_to_predicted_perf
import matchmaker.matchmaker as _mm_mod

FRAME_RATE = 30


def run_with_feature(piece, performer, algorithm='dixon', use_cqt=False, use_resynth=False):
    """Run score following with optional CQT features and resynth."""
    piece_dir = piece["asap_dir"]
    xml_path = os.path.join(piece_dir, "xml_score.musicxml")
    wav_path = os.path.join(piece_dir, f"{performer}.wav")
    if not os.path.exists(wav_path):
        return None

    initial_tempo = _get_xml_tempo(xml_path)
    tempo_for_eval = initial_tempo

    # If resynth: get global ratio from pass 1 (always with STFT for speed)
    if use_resynth:
        _mm_mod.DEFAULT_TEMPO = initial_tempo
        mm0 = Matchmaker(score_file=xml_path, performance_file=wav_path,
                        input_type="audio", wait=False)
        na0 = mm0.score_part.note_array()
        fb0 = na0["onset_beat"].min()
        trim0 = max(fb0, 0) * (60.0 / initial_tempo)
        ts0 = int(trim0 * SAMPLE_RATE)
        if ts0 > 0:
            mm0.score_audio = mm0.score_audio[ts0:]
            mm0.reference_features = mm0.processor(mm0.score_audio)
        mm0.score_follower = OnlineTimeWarpingDixon(
            reference_features=mm0.reference_features, queue=mm0.stream.queue)
        for _ in mm0.run(verbose=False):
            pass
        wp0 = np.array(mm0.score_follower.warping_path)
        if wp0[1][-1] > 0:
            ratio = wp0[0][-1] / wp0[1][-1]
            tempo_for_eval = initial_tempo * ratio

    # Main run
    _mm_mod.DEFAULT_TEMPO = tempo_for_eval
    mm = Matchmaker(score_file=xml_path, performance_file=wav_path,
                    input_type="audio", wait=False)

    na = mm.score_part.note_array()
    first_beat = na["onset_beat"].min()
    bt = mm.score_part.time_sigs[0].beat_type if mm.score_part.time_sigs else 4
    trim = max(first_beat, 0) * (60.0 / tempo_for_eval)
    ts = int(trim * SAMPLE_RATE)
    if ts > 0:
        mm.score_audio = mm.score_audio[ts:]

    # Swap processor if CQT
    if use_cqt:
        # Reference: one-shot CQT on the synthesized score audio.
        mm.processor = CQTChromagramProcessor(sample_rate=SAMPLE_RATE)
        mm.reference_features = mm.processor(mm.score_audio)
        # Stream: precompute CQT on the performance audio up front and hand
        # out frames k-th at a time. Avoids per-chunk CQT (slow, noisy).
        import librosa as _lr
        perf_y, _ = _lr.load(wav_path, sr=SAMPLE_RATE)
        mm.stream.processor = StreamCQTChromagramProcessor(perf_y, sample_rate=SAMPLE_RATE)
    else:
        mm.reference_features = mm.processor(mm.score_audio)

    # Create score follower
    if algorithm == 'dixon':
        mm.score_follower = OnlineTimeWarpingDixon(
            reference_features=mm.reference_features, queue=mm.stream.queue)
    else:
        mm.score_follower = OnlineTimeWarpingArzt(
            reference_features=mm.reference_features, queue=mm.stream.queue,
            frame_rate=mm.frame_rate)

    for _ in mm.run(verbose=False):
        pass
    wp = np.array(mm.score_follower.warping_path)

    # Evaluate
    gt_pt, gt_bt = load_gt(piece_dir, performer)
    sp_bmin = max(na["onset_beat"].min(), 0.0)
    sp_bmax = na["onset_beat"].max()
    lut_b = np.linspace(sp_bmin, sp_bmax, 2000)
    lut_t = _build_score_times_for_beats(mm.score_part, lut_b, tempo_for_eval)
    gt_beats_scaled = gt_bt * (bt / 4)
    score_annots = np.interp(gt_beats_scaled, lut_b, lut_t) - trim
    ref_dur = len(mm.score_audio) / SAMPLE_RATE
    score_annots = np.clip(score_annots, 0, ref_dur)

    pred = transfer_from_score_to_predicted_perf(wp, score_annots, FRAME_RATE)
    ok = ~np.isnan(pred)
    total = len(gt_pt)
    errors = np.abs(gt_pt[ok] - pred[ok]) * 1000.0
    return float(np.sum(errors <= 500) / total * 100)


def main():
    results = []

    for piece in PIECES:
        pid = piece["id"]
        performers = piece["performers"]
        print(f"\n  {piece['name']} ({len(performers)} perfs)")

        for perf in performers:
            t0 = time.time()
            row = {'piece': pid, 'performer': perf}

            for label, kwargs in [
                ('dixon_stft', dict(algorithm='dixon', use_cqt=False, use_resynth=False)),
                ('dixon_cqt', dict(algorithm='dixon', use_cqt=True, use_resynth=False)),
                ('dixon_cqt_resy', dict(algorithm='dixon', use_cqt=True, use_resynth=True)),
            ]:
                try:
                    row[label] = run_with_feature(piece, perf, **kwargs)
                except Exception as e:
                    row[label] = float('nan')

            results.append(row)
            s = row.get('dixon_stft', 0) or 0
            c = row.get('dixon_cqt', 0) or 0
            cr = row.get('dixon_cqt_resy', 0) or 0
            print(f"    {perf:20s} stft={s:5.1f}  cqt={c:5.1f}({c-s:+.1f})  cqt+r={cr:5.1f}({cr-s:+.1f})  [{time.time()-t0:.0f}s]")

    df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "exp_cqt_native.csv")
    df.to_csv(out_path, index=False)

    print(f"\n  SUMMARY ({len(df)} pairs)")
    v = df.dropna(subset=['dixon_stft', 'dixon_cqt'])
    print(f"  Dixon+STFT:      {v['dixon_stft'].mean():.1f}%")
    print(f"  Dixon+CQT:       {v['dixon_cqt'].mean():.1f}% ({v['dixon_cqt'].mean()-v['dixon_stft'].mean():+.1f}pp)")
    v2 = df.dropna(subset=['dixon_stft', 'dixon_cqt_resy'])
    if len(v2) > 0:
        print(f"  Dixon+CQT+Resy:  {v2['dixon_cqt_resy'].mean():.1f}% ({v2['dixon_cqt_resy'].mean()-v2['dixon_stft'].mean():+.1f}pp)")


if __name__ == "__main__":
    main()
