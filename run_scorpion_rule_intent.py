"""
Rule-based section intent baseline for Ballade 1.

Extracts ALL tempo markings from MusicXML using partitura (ConstantTempoDirection,
IncreasingTempoDirection, DecreasingTempoDirection, Fermata) and maps them to a
piecewise γ(b) using the same Italian → log-tempo table our v3_rich data prep uses.

Compares this rule-based γ to the cached LLM γ (1 marking, 2 sections) on the
n=6 Ballade 1 LLM-eval subset under Arzt+STFT.

Honest test: if the rule-based richer γ gives \\approx the same +16.7pp gain over
baseline as cached LLM γ, then the +47pp in tab:intent(b) is not LLM-magic but
"any text-derived piecewise γ with the same global slowdown".
If rule γ gives substantially MORE gain, then the paper's intent claim
strengthens (richer markings help) and we should release the rule-derived γ
as a stronger reproducible artifact.
"""
from __future__ import annotations
import os, pickle, sys, tempfile, time, warnings, json
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import partitura as pt
from partitura.score import (
    ConstantTempoDirection, DecreasingTempoDirection, IncreasingTempoDirection,
    ConstantLoudnessDirection, Fermata,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_full_experiment import PIECES, _get_xml_tempo, load_gt, run_single
import matchmaker.matchmaker as _mm_mod

PIECE_ID = 'chopin_ballade1'
PERFORMERS = ['BuiJL04M', 'JIA06M', 'KimSuyeon04M',
              'Richardson13M', 'Sladek04M', 'Zhou06M']
OUT_CSV = 'results/scorpion_rule_intent.csv'

# Same Italian → log-tempo lookup as prepare_data_v3_rich.py
# Log-tempo offsets (gamma = exp(offset)) derived from standard BPM
# midpoints in The New Grove Dictionary of Music and Musicians (2nd ed.,
# 2001) and The Harvard Dictionary of Music (4th ed., Randel 2003).
# Baseline = Moderato midpoint 114 BPM. log_offset = ln(bpm_mid / 114).
TEMPO_TABLE = {
    # Absolute tempo markings (Grove/Harvard midpoints)
    'grave':        -1.18,  # 35 BPM (range 25-45)
    'largo':        -0.82,  # 50 BPM (40-60)
    'lento':        -0.78,  # 52 BPM (45-60)
    'largamente':   -0.82,
    'larghetto':    -0.59,  # 63 BPM (60-66)
    'adagio':       -0.47,  # 71 BPM (66-76)
    'adagietto':    -0.42,  # 75 BPM (70-80)
    'andante':      -0.21,  # 92 BPM (76-108)
    'andantino':    -0.19,  # 94 BPM
    'moderato':      0.00,  # 114 BPM (108-120) baseline
    'allegretto':    0.02,  # 116 BPM (112-120)
    'allegro':       0.19,  # 138 BPM (120-156)
    'allegro moderato': 0.12,
    'allegro non troppo': 0.14,
    'vivace':        0.38,  # 166 BPM (156-176)
    'vivo':          0.38,
    'presto':        0.48,  # 184 BPM (168-200)
    'prestissimo':   0.58,  # 204 BPM (200-208+)
    'a tempo':       0.00,
    'tempo primo':   0.00,
    'tempo i':       0.00,
    # Relative markings (conventional ratios from music theory)
    'meno mosso':   -0.26,  # 0.77x preceding (typical 0.70-0.85)
    'piu mosso':     0.20,  # 1.22x
    'più mosso':     0.20,
    'agitato':       0.26,  # 1.30x
    'con moto':      0.10,  # 1.10x
    'sostenuto':    -0.11,  # 0.90x
    'tranquillo':   -0.11,  # 0.90x
    'appassionato':  0.14,  # 1.15x
    'animato':       0.18,  # 1.20x
    'maestoso':     -0.16,  # 0.85x
    'cantabile':     0.00,  # style marking (no tempo change)
}


def lookup_tempo(text: str, default: float = 0.0) -> float:
    if not text: return default
    t = text.strip().lower()
    if t in TEMPO_TABLE: return TEMPO_TABLE[t]
    for k in sorted(TEMPO_TABLE.keys(), key=len, reverse=True):
        if t.startswith(k):
            return TEMPO_TABLE[k]
    return default


def extract_rule_gamma(xml_path: str, max_beat: int) -> tuple[np.ndarray, list]:
    """Build piecewise-constant γ(b) from ALL MusicXML tempo markings.
    Returns (gamma array length max_beat, list of (beat, text, gamma) markings).
    """
    score = pt.load_musicxml(xml_path)
    part = score.parts[0]
    bm = part.beat_map

    markings = []
    for d in part.iter_all(ConstantTempoDirection):
        txt = getattr(d, 'raw_text', '') or getattr(d, 'text', '')
        if d.start is None: continue
        beat = float(bm(d.start.t))
        log_t = lookup_tempo(txt, 0.0)
        gamma = float(np.exp(log_t))  # log-ratio → multiplicative γ
        markings.append((beat, 'tempo', txt, gamma))

    # Increase / decrease tempo directions: treat as gradual ramps
    for cls, sign in [(IncreasingTempoDirection, +0.3),
                       (DecreasingTempoDirection, -0.3)]:
        for d in part.iter_all(cls):
            if d.start is None: continue
            beat = float(bm(d.start.t))
            txt = getattr(d, 'raw_text', '') or getattr(d, 'text', '')
            # For accel./rall., scale γ by sign (relative)
            markings.append((beat, 'change', txt, np.exp(sign)))

    markings.sort(key=lambda x: x[0])

    # Build γ(b) piecewise-constant: between markings, hold last value
    gamma = np.ones(max_beat, dtype=np.float32)
    cur = 1.0
    mi = 0
    for b in range(max_beat):
        while mi < len(markings) and markings[mi][0] <= b:
            cur = markings[mi][3]
            mi += 1
        gamma[b] = cur

    # Apply piece-level prior: many Chopin pieces are played slower than XML's
    # marked tempo. Calibrate against the cached LLM γ that successfully gave
    # the +47pp result on Ballade 1 (mean γ ≈ 0.75 = 25% slower).
    # Apply the same mean shift to the rule γ if its mean is far from the
    # known-good calibration.
    return gamma, markings


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
    piece = next(p for p in PIECES if p['id'] == PIECE_ID)
    xml_path = os.path.join(piece['asap_dir'], 'xml_score.musicxml')
    tempo = _get_xml_tempo(xml_path)

    # First find max beat from any performer's annotation
    _, gt_bt = load_gt(piece['asap_dir'], PERFORMERS[0])
    max_beat = int(gt_bt.max()) + 1

    gamma_rule, markings = extract_rule_gamma(xml_path, max_beat)
    print(f'Rule γ markings extracted: {len(markings)}')
    for beat, kind, txt, g in markings[:25]:
        print(f'  beat {beat:6.1f}  {kind:7s}  γ={g:.3f}  "{txt}"')
    print(f'γ stats: min={gamma_rule.min():.3f} mean={gamma_rule.mean():.3f} '
          f'max={gamma_rule.max():.3f} std={gamma_rule.std():.3f}')
    print(f'Distinct γ values: {len(np.unique(np.round(gamma_rule, 3)))}')

    # Also load cached LLM γ for comparison
    llm_data = json.load(open('/home/ykim3098/scorpion-demo/assets/llm_gamma_all.json'))[PIECE_ID]
    gamma_llm = np.ones(max_beat, dtype=np.float32)
    for s in llm_data['sections']:
        s0, e0, g_s, g_e = s
        lo, hi = int(round(s0)), int(round(e0))
        hi = min(hi, max_beat)
        if hi > lo:
            gamma_llm[lo:hi] = np.linspace(g_s, g_e, hi - lo, endpoint=False)
    print(f'LLM γ stats: min={gamma_llm.min():.3f} mean={gamma_llm.mean():.3f} std={gamma_llm.std():.3f}')

    rows = []
    for perf in PERFORMERS:
        # Baseline (none)
        _mm_mod.DEFAULT_TEMPO = tempo
        t0 = time.time()
        r_base = run_single(piece, perf, 'xml', algorithm='arzt')
        rows.append({'perf': perf, 'cond': 'none', 'ar_500': r_base.get('ar_500'),
                     'wall_s': round(time.time()-t0,1)})
        print(f'  {perf}  none      AR={r_base.get("ar_500", "NaN")}')

        # Rule γ
        t0 = time.time()
        r_rule = inject_and_run(piece, perf, tempo, gamma_rule, max_beat)
        rows.append({'perf': perf, 'cond': 'rule_intent', 'ar_500': r_rule.get('ar_500'),
                     'wall_s': round(time.time()-t0,1)})
        print(f'  {perf}  rule      AR={r_rule.get("ar_500", "NaN")}')

        # LLM γ (cached, 2-section)
        t0 = time.time()
        r_llm = inject_and_run(piece, perf, tempo, gamma_llm, max_beat)
        rows.append({'perf': perf, 'cond': 'llm_cached', 'ar_500': r_llm.get('ar_500'),
                     'wall_s': round(time.time()-t0,1)})
        print(f'  {perf}  llm_cache AR={r_llm.get("ar_500", "NaN")}')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nWrote {len(df)} rows to {OUT_CSV}')
    print()
    print(df.groupby('cond')['ar_500'].agg(['mean', 'median', 'count']).round(2))


if __name__ == '__main__':
    main()
