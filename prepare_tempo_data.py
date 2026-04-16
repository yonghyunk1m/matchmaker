"""
Prepare tempo trajectory data for prediction model
====================================================
Extract per-beat tempo profiles from ALL ASAP performances.
Output: dataset of (piece_id, tempo_trajectory) pairs for training.
"""
import json
import os
import numpy as np
import pickle

ASAP_JSON = 'datasets/asap-dataset/asap_annotations.json'
OUTPUT_PATH = 'output/tempo_trajectories.pkl'


def extract_tempo_profile(perf_beats, score_beats):
    """Extract per-beat tempo ratio from performance."""
    pt = np.array(perf_beats, dtype=np.float64)
    bt = np.array(score_beats, dtype=np.float64)

    if len(pt) < 5:
        return None

    # Align lengths
    min_len = min(len(pt), len(bt))
    pt = pt[:min_len]
    bt = bt[:min_len]

    ioi = np.diff(pt)  # inter-onset intervals (seconds)
    bd = np.diff(bt)   # beat distances
    ioi = np.maximum(ioi, 0.001)
    bd = np.maximum(bd, 0.001)

    # Local tempo: beats per second at each beat
    local_tempo = bd / ioi

    # Normalize by median → tempo ratio
    median_tempo = np.median(local_tempo)
    if median_tempo <= 0:
        return None
    tempo_ratios = local_tempo / median_tempo

    # Clip extreme values
    tempo_ratios = np.clip(tempo_ratios, 0.1, 10.0)

    return {
        'beat_positions': bt[:-1],
        'tempo_ratios': tempo_ratios,
        'median_tempo': median_tempo,
        'n_beats': len(tempo_ratios),
    }


def main():
    with open(ASAP_JSON) as f:
        asap = json.load(f)

    dataset = []
    piece_counts = {}

    for key, val in asap.items():
        if 'performance_beats' not in val or 'midi_score_beats' not in val:
            continue

        piece_id = '/'.join(key.split('/')[:3])
        performer = key.split('/')[-1].replace('.mid', '')

        profile = extract_tempo_profile(
            val['performance_beats'],
            val['midi_score_beats']
        )

        if profile is None:
            continue

        profile['piece_id'] = piece_id
        profile['performer'] = performer
        dataset.append(profile)

        piece_counts[piece_id] = piece_counts.get(piece_id, 0) + 1

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(dataset, f)

    print(f'Total profiles: {len(dataset)}')
    print(f'Unique pieces: {len(piece_counts)}')
    print(f'Avg performers per piece: {np.mean(list(piece_counts.values())):.1f}')

    # Stats
    lengths = [d['n_beats'] for d in dataset]
    print(f'Beat sequence length: min={min(lengths)}, max={max(lengths)}, '
          f'median={np.median(lengths):.0f}')

    # Example
    ex = dataset[0]
    print(f'\nExample: {ex["piece_id"]}/{ex["performer"]}')
    print(f'  Beats: {ex["n_beats"]}, Median tempo: {ex["median_tempo"]:.2f} bps')
    print(f'  Tempo ratios (first 10): {np.round(ex["tempo_ratios"][:10], 3)}')


if __name__ == '__main__':
    main()
