"""Downstream eval using LLM's markings_annotated as piecewise-CONSTANT γ
(matches rule's output format; isolates whether LLM's ramps are the problem).
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, tempfile, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_full_experiment import PIECES, _get_xml_tempo, load_gt, run_single
import matchmaker.matchmaker as _mm_mod

PIECE_ID = 'chopin_ballade1'
PERFORMERS = ['BuiJL04M', 'JIA06M', 'KimSuyeon04M',
              'Richardson13M', 'Sladek04M', 'Zhou06M']


def build_pwc_gamma(markings: list, max_beat: int) -> np.ndarray:
    """Piecewise-constant γ built from markings_annotated (ignore section ramps).
    Between two markings, hold the earlier marking's γ until the next arrives."""
    g = np.ones(max_beat, dtype=np.float32)
    cur = 1.0
    ordered = sorted(markings, key=lambda m: m['beat'])
    mi = 0
    for b in range(max_beat):
        while mi < len(ordered) and ordered[mi]['beat'] <= b:
            cur = float(ordered[mi]['gamma'])
            mi += 1
        g[b] = cur
    return np.clip(g, 0.3, 3.0)


def inject_and_run(piece, perf, tempo, gamma, max_beat):
    plan_path = tempfile.mktemp(suffix='.pkl')
    arr = np.ones(max_beat, dtype=np.float32)
    for b in range(min(len(gamma), max_beat)):
        arr[b] = float(np.clip(gamma[b], 0.3, 3.0))
    locked = np.zeros(max_beat, dtype=bool)
    with open(plan_path, 'wb') as f:
        pickle.dump((arr, arr, locked), f)
    _mm_mod.DEFAULT_TEMPO = tempo
    r = run_single(piece, perf, 'xml', algorithm='arzt',
                   plan_path_override=plan_path)
    os.remove(plan_path)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--llm-output', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rec = json.load(open(args.llm_output))
    resp = rec['parsed_response']
    marks = resp['markings_annotated']
    print(f'{len(marks)} markings from LLM')

    piece = next(p for p in PIECES if p['id'] == PIECE_ID)
    xml_path = os.path.join(piece['asap_dir'], 'xml_score.musicxml')
    tempo = _get_xml_tempo(xml_path)
    _, gt_bt = load_gt(piece['asap_dir'], PERFORMERS[0])
    max_beat = int(gt_bt.max()) + 1
    gamma = build_pwc_gamma(marks, max_beat)
    print(f'PWC γ stats: min={gamma.min():.3f} mean={gamma.mean():.3f} '
          f'max={gamma.max():.3f} std={gamma.std():.3f}')

    rows = []
    for perf in PERFORMERS:
        t0 = time.time()
        r_llm = inject_and_run(piece, perf, tempo, gamma, max_beat)
        rows.append({'perf': perf, 'cond': 'llm_v1_pwc', 'ar_500': r_llm.get('ar_500'),
                     'wall_s': round(time.time()-t0, 1)})
        print(f'  {perf}  llm_v1_pwc  AR={r_llm.get("ar_500")}')

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'\nWrote {len(df)} rows to {args.out}')
    print(df.groupby('cond')['ar_500'].agg(['mean', 'median', 'count']).round(2))


if __name__ == '__main__':
    main()
