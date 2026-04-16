"""SCORPION LLM γ generation — reproducible caller.

Reads a prompt template from `llm/prompts/`, fills in MusicXML-extracted
markings (and optionally a performer's prior rehearsal γ), calls Claude via the
Anthropic API at temperature=0, and writes the response + full call metadata to
`llm/outputs/<piece_id>_<prompt_version>_<timestamp>.json`.

Every call is versioned. Nothing is overwritten. The prompt template path, the
SHA256 of the prompt contents, the input JSON, the model id, and the raw
response are all persisted.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python llm/call_llm.py --piece chopin_ballade1 --prompt score_only_v1
    python llm/call_llm.py --piece chopin_ballade1 --prompt performer_aware_v1 \\
        --performer BuiJL04M --prior-run results/wp_BuiJL04M_pass1.pkl

Requires `pip install anthropic partitura`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = ROOT / 'llm' / 'prompts'
OUTPUT_DIR = ROOT / 'llm' / 'outputs'
ASAP_DIR = ROOT / 'datasets' / 'asap-dataset'

PIECE_XML = {
    'chopin_ballade1': ASAP_DIR / 'Chopin' / 'Ballades' / '1' / 'xml_score.musicxml',
    'chopin_ballade2': ASAP_DIR / 'Chopin' / 'Ballades' / '2' / 'xml_score.musicxml',
    'chopin_ballade3': ASAP_DIR / 'Chopin' / 'Ballades' / '3' / 'xml_score.musicxml',
    'chopin_ballade4': ASAP_DIR / 'Chopin' / 'Ballades' / '4' / 'xml_score.musicxml',
}

MODEL_ID = 'claude-opus-4-5-20250929'  # pinned; exact id matters for reproducibility
MAX_TOKENS = 8000
TEMPERATURE = 0.0


def extract_markings(xml_path: Path) -> tuple[list[dict], float]:
    import warnings
    warnings.filterwarnings('ignore')
    import partitura as pt
    from partitura.score import (
        ConstantTempoDirection, IncreasingTempoDirection,
        DecreasingTempoDirection, Fermata,
    )

    score = pt.load_musicxml(str(xml_path))
    part = score.parts[0]
    bm = part.beat_map

    marks = []
    for cls, kind in [
        (ConstantTempoDirection, 'tempo'),
        (IncreasingTempoDirection, 'accel'),
        (DecreasingTempoDirection, 'rall'),
    ]:
        for d in part.iter_all(cls):
            if d.start is None:
                continue
            txt = getattr(d, 'raw_text', '') or getattr(d, 'text', '') or ''
            marks.append({
                'beat': round(float(bm(d.start.t)), 2),
                'text': txt.strip(),
                'kind': kind,
            })
    for d in part.iter_all(Fermata):
        if d.start is None:
            continue
        marks.append({
            'beat': round(float(bm(d.start.t)), 2),
            'text': 'fermata',
            'kind': 'fermata',
        })

    marks.sort(key=lambda m: m['beat'])

    # Score beat max = last note end across all notes in the part
    ends = []
    for n in part.notes_tied:
        if n.end is not None:
            ends.append(float(bm(n.end.t)))
    if not ends:
        for n in part.iter_all(pt.score.GenericNote):
            if getattr(n, 'end', None) is not None:
                ends.append(float(bm(n.end.t)))
    max_beat = float(max(ends)) if ends else float(marks[-1]['beat'] if marks else 0.0)
    return marks, round(max_beat, 2)


def build_input_json(piece_id: str, markings: list[dict], max_beat: float,
                     performer_id: str | None = None,
                     prior_pass: dict | None = None) -> dict:
    payload = {
        'piece_id': piece_id,
        'score_beat_max': max_beat,
        'markings': markings,
    }
    if performer_id is not None:
        payload['performer_id'] = performer_id
    if prior_pass is not None:
        payload['prior_pass_ar500'] = prior_pass['ar500']
        payload['prior_rehearsal_gamma'] = prior_pass['gamma_decimated']
    return payload


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def call_anthropic(prompt: str) -> dict:
    try:
        import anthropic
    except ImportError:
        raise SystemExit(
            'ERROR: `anthropic` package not installed.\n'
            'Run: pip install anthropic\n'
            'Then set: export ANTHROPIC_API_KEY=sk-ant-...'
        )
    client = anthropic.Anthropic()
    t0 = time.time()
    resp = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{'role': 'user', 'content': prompt}],
    )
    elapsed = round(time.time() - t0, 2)
    return {
        'backend': 'anthropic_sdk',
        'model': MODEL_ID,
        'temperature': TEMPERATURE,
        'max_tokens': MAX_TOKENS,
        'elapsed_s': elapsed,
        'usage': {
            'input_tokens': resp.usage.input_tokens,
            'output_tokens': resp.usage.output_tokens,
        },
        'content': resp.content[0].text if resp.content else '',
    }


def call_claude_cli(prompt: str, binary: str = 'claude') -> dict:
    """Fallback caller via `claude -p` (Claude Code CLI, non-interactive).
    Reuses the user's existing Claude Code auth; no API key needed."""
    import subprocess
    t0 = time.time()
    proc = subprocess.run([binary, '-p'],
                          input=prompt, text=True, capture_output=True,
                          timeout=900)
    elapsed = round(time.time() - t0, 2)
    if proc.returncode != 0:
        raise RuntimeError(f'claude -p failed: {proc.stderr}')
    return {
        'backend': 'claude_cli',
        'binary': binary,
        'model': 'claude-code-default',
        'temperature': TEMPERATURE,
        'elapsed_s': elapsed,
        'stderr': proc.stderr,
        'content': proc.stdout,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--piece', required=True, choices=list(PIECE_XML.keys()))
    ap.add_argument('--prompt', required=True,
                    help='prompt filename stem (e.g. score_only_v1)')
    ap.add_argument('--performer', default=None)
    ap.add_argument('--prior-run', default=None,
                    help='pickle of prior pass warping path (for performer_aware)')
    ap.add_argument('--dry-run', action='store_true',
                    help='skip API call; just write the rendered prompt')
    ap.add_argument('--backend', choices=['sdk', 'cli'], default='sdk',
                    help='sdk=anthropic SDK (needs ANTHROPIC_API_KEY); '
                         'cli=claude -p (reuses Claude Code auth)')
    ap.add_argument('--claude-bin', default='claude',
                    help='path to claude binary (for --backend cli)')
    args = ap.parse_args()

    prompt_path = PROMPT_DIR / f'{args.prompt}.md'
    if not prompt_path.exists():
        raise SystemExit(f'Prompt not found: {prompt_path}')
    prompt_text = prompt_path.read_text()
    prompt_hash = sha256(prompt_text)

    xml_path = PIECE_XML[args.piece]
    marks, max_beat = extract_markings(xml_path)
    print(f'[{args.piece}] extracted {len(marks)} markings over {max_beat} beats')

    prior = None
    if args.prior_run is not None:
        import pickle
        with open(args.prior_run, 'rb') as f:
            pass_data = pickle.load(f)
        # Decimate the prior γ trajectory to ≤ 200 samples
        import numpy as np
        g = np.asarray(pass_data['gamma_per_beat'], dtype=float)
        stride = max(1, len(g) // 200)
        prior = {
            'ar500': float(pass_data['ar500']),
            'gamma_decimated': [[i * stride, float(g[i * stride])]
                                for i in range(len(g) // stride)],
        }

    input_payload = build_input_json(args.piece, marks, max_beat,
                                      args.performer, prior)
    input_json_str = json.dumps(input_payload, indent=2, ensure_ascii=False)
    filled_prompt = prompt_text.replace('{{INPUT_JSON}}', input_json_str)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    tag = f'{args.piece}_{args.prompt}'
    if args.performer is not None:
        tag += f'_{args.performer}'
    out_path = OUTPUT_DIR / f'{tag}_{stamp}.json'

    record = {
        'piece_id': args.piece,
        'prompt_file': str(prompt_path.relative_to(ROOT)),
        'prompt_sha256': prompt_hash,
        'input_json': input_payload,
        'rendered_prompt': filled_prompt,
        'timestamp_utc': stamp,
    }

    if args.dry_run:
        print(f'[dry-run] would call model, writing rendered prompt only')
        record['mode'] = 'dry_run'
    else:
        if args.backend == 'sdk':
            api_result = call_anthropic(filled_prompt)
        else:
            api_result = call_claude_cli(filled_prompt, binary=args.claude_bin)
        record['mode'] = 'live'
        record['api_result'] = api_result
        # Try to parse content as JSON (strip possible markdown fences)
        text = api_result['content'].strip()
        if text.startswith('```'):
            text = text.split('```', 2)[1]
            if text.startswith('json'):
                text = text[4:]
            text = text.rsplit('```', 1)[0].strip()
        try:
            record['parsed_response'] = json.loads(text)
        except Exception as e:
            record['parse_error'] = str(e)

    with open(out_path, 'w') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f'wrote {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
