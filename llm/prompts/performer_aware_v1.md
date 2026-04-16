# SCORPION LLM Intent — Performer-Aware Prompt (v1)

**Version**: v1 (2026-04-15)
**Purpose**: Same contract as `score_only_v1.md`, but additionally conditioned on
the performer's prior rehearsal tempo trajectory. This tests whether feeding
fine-grained performer history to the LLM refines $\gamma(b)$ beyond what the
score alone can supply.

**Additional input (on top of score-only prompt)**:
- `PERFORMER_ID`: string identifier (e.g. `BuiJL04M`)
- `PRIOR_REHEARSAL_GAMMA`: a decimated (≤ 200 samples) list of
  `[beat, gamma_observed]` pairs obtained from this performer's prior
  warping-path run on the same piece. Represents the *empirical* tempo the
  performer has taken in a previous rehearsal pass.
- `PRIOR_PASS_AR500`: the AR@500 achieved on that prior pass (float, 0–100).

**Decision policy** (explicit, for auditability):

1. If `PRIOR_PASS_AR500 ≥ 60`, the prior run tracked well; trust the empirical
   γ and align the output sections to the decimated samples, smoothing only
   around score markings.
2. If `PRIOR_PASS_AR500 < 20`, the prior run failed; ignore the prior γ entirely
   and fall back to the score-only prompt's output.
3. Otherwise (borderline tracking), blend: for each section, use a 60/40 weighted
   average of (empirical γ from decimation) and (score-lookup γ). Respect the
   lookup constants as the *upper/lower envelope* — never output γ outside the
   range set by the surrounding markings' constants scaled by ±30%.

**Output contract**: identical to `score_only_v1.md`, plus a new top-level field:
```json
"performer_id": "<string>",
"prior_pass_ar500": <float>,
"blending_mode": "<score_only|empirical|blend>"
```

**Reproducibility requirement**: the `blending_mode` decision MUST be
deterministic given the inputs. No random choices.

**Use case**: this is the *conversational rehearsal partner* setting — after one
or more rehearsal passes, the LLM refines the next pass's γ using the performer's
history. This is the evidence for SCORPION's 4th contribution axis (intent that
incorporates performer-specific rehearsal signal, not just score text).

---

## Input data block (filled at call time)

```json
{{INPUT_JSON}}
```

Produce ONLY the output JSON. No commentary, no Markdown fencing.
