# SCORPION LLM Intent — Reproducibility Directory

This directory holds every artifact needed to reproduce the LLM-derived
$\gamma(b)$ used in the paper's intent-injection experiments.

## Layout

- `prompts/` — versioned prompt templates (`score_only_v1.md`,
  `performer_aware_v1.md`, ...). Every call references exactly one file.
- `outputs/` — one JSON per LLM call. Filename encodes piece, prompt version,
  optional performer, and UTC timestamp. Each record contains:
  - `prompt_file`, `prompt_sha256` — proves which prompt was sent
  - `input_json` — the filled-in data the LLM saw
  - `rendered_prompt` — the exact string sent to the API
  - `api_result.model` — pinned model id
  - `api_result.content` — raw response
  - `parsed_response` — JSON-parsed γ sections (when valid)
- `call_llm.py` — the caller. Deterministic at `temperature=0`.
- `README.md` — this file.

## Reproducing a γ from scratch

```bash
pip install anthropic partitura
export ANTHROPIC_API_KEY=sk-ant-...

# Score-only γ for Ballade 1
python llm/call_llm.py --piece chopin_ballade1 --prompt score_only_v1

# Performer-aware γ conditioning on a prior rehearsal pass
python llm/call_llm.py --piece chopin_ballade1 --prompt performer_aware_v1 \\
    --performer BuiJL04M --prior-run results/wp_BuiJL04M_pass1.pkl
```

## Paper numbers → files

| Paper claim | Output file |
|---|---|
| tab:llm_intent "LLM section" 26.7% on Ballade 1 | TBD — see `outputs/` after first live call |
| fig:markings_to_gamma "LLM γ" panel | same |
| tab:intent(b) `+LLM` column | same |

## Historical note (2026-04-15 audit)

The original paper results for `llm` on Ballade 1 (AR@500 = 26.67 in
`results/exp_a_arzt_cqt_intent.csv`) were produced by an ad-hoc LLM path whose
exact prompt was not preserved. The cached JSON at
`scorpion-demo/assets/llm_gamma_all.json` produces only AR@500 = 10.08 because
it captures a single marking (`a tempo (meno mosso)`) with a flat
γ = 0.75 across the entire Ballade. This asymmetry was detected during the
pre-submission review and is the reason this reproducibility directory exists:
from now on, every LLM-derived γ ships with its full prompt, input, and model
version for audit.
