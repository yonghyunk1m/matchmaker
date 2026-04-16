"""Downstream evaluation: inject a newly-generated LLM γ (from llm/outputs/) into
Arzt+STFT on Ballade 1 (n=6) and log AR@500 per performer.

Usage:
    python llm/run_downstream.py \
        --llm-output llm/outputs/chopin_ballade1_score_only_v1_<stamp>.json \
        --out results/scorpion_llm_v1.csv
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


def build_gamma_from_sections(sections: list, max_beat: int) -> np.ndarray:
    """Build piecewise-linear γ(b) array from section tuples
    [start_beat, end_beat, gamma_start, gamma_end]."""
    gamma = np.ones(max_beat, dtype=np.float32)
    for s in sections:
        s0, e0, g_s, g_e = s
        lo = max(0, int(round(s0)))
        hi = min(max_beat, int(round(e0)))
        if hi <= lo:
            continue
        n = hi - lo
        gamma[lo:hi] = np.linspace(g_s, g_e, n, endpoint=False)
    # Clip to valid range
    return np.clip(gamma, 0.3, 3.0)


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
    sections = resp['sections']
    piece_id = resp['piece_id']
    assert piece_id == PIECE_ID, f'Expected {PIECE_ID}, got {piece_id}'
    print(f'Loaded LLM γ from {args.llm_output}')
    print(f'  sections: {len(sections)}')
    print(f'  prompt: {rec["prompt_file"]}  sha256: {rec["prompt_sha256"][:12]}')
    print(f'  backend: {rec["api_result"].get("backend", "?")}')
    # Derive cond label from prompt file (e.g. 'llm_score_only_v2')
    prompt_stem = rec['prompt_file'].replace('llm/prompts/', '').replace('.md', '')
    cond_label = f'llm_{prompt_stem}'

    piece = next(p for p in PIECES if p['id'] == PIECE_ID)
    xml_path = os.path.join(piece['asap_dir'], 'xml_score.musicxml')
    tempo = _get_xml_tempo(xml_path)

    # Establish max_beat from first performer's annotation
    _, gt_bt = load_gt(piece['asap_dir'], PERFORMERS[0])
    max_beat = int(gt_bt.max()) + 1
    gamma = build_gamma_from_sections(sections, max_beat)
    print(f'γ stats: min={gamma.min():.3f} mean={gamma.mean():.3f} '
          f'max={gamma.max():.3f} std={gamma.std():.3f}')

    rows = []
    for perf in PERFORMERS:
        # Baseline
        _mm_mod.DEFAULT_TEMPO = tempo
        t0 = time.time()
        r_base = run_single(piece, perf, 'xml', algorithm='arzt')
        base_ar = r_base.get('ar_500')
        rows.append({'perf': perf, 'cond': 'none', 'ar_500': base_ar,
                     'wall_s': round(time.time()-t0, 1)})
        print(f'  {perf}  none  AR={base_ar}')
        # LLM γ
        t0 = time.time()
        r_llm = inject_and_run(piece, perf, tempo, gamma, max_beat)
        llm_ar = r_llm.get('ar_500')
        rows.append({'perf': perf, 'cond': cond_label, 'ar_500': llm_ar,
                     'wall_s': round(time.time()-t0, 1)})
        print(f'  {perf}  {cond_label}  AR={llm_ar}')

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'\nWrote {len(df)} rows to {args.out}')
    print(df.groupby('cond')['ar_500'].agg(['mean', 'median', 'count']).round(2))


if __name__ == '__main__':
    main()
