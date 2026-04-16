# SCORPION LLM Intent — Score-Only Prompt (v2, piecewise-constant)

**Version**: v2 (2026-04-15)
**Diff vs v1**: enforces piecewise-CONSTANT γ between markings. No ramps.
The tempo ratio jumps at each marking and holds flat until the next marking.
This matches the rule baseline's output format and is what the OLTW tracker's
step-size modulation actually consumes (ramps destabilize the accumulated cost).

**Purpose**: Convert a MusicXML's printed Italian tempo/character markings into a
per-beat tempo-ratio trajectory $\gamma(b) \in (0, 5]$ that represents the
performer's *expected* intent before performance begins.

**Input**: same as v1 (`PIECE_ID`, `SCORE_BEAT_MAX`, `MARKINGS`).

**Output contract** (strict JSON schema):
```json
{
  "piece_id": "<string>",
  "score_beat_max": <float>,
  "prompt_version": "score_only_v2",
  "markings_annotated": [
    {
      "beat": <float>,
      "text": "<string>",
      "kind": "<tempo|accel|rall|fermata>",
      "gamma": <float>,
      "reasoning": "<string, one short sentence>"
    }
  ],
  "sections": [
    [<start_beat>, <end_beat>, <gamma>, <gamma>]
  ]
}
```

**Key invariant**: for every section, `gamma_start == gamma_end`. No ramps.
A section whose two γ values differ by more than $10^{-6}$ is a specification
violation and the output is invalid.

**Lookup table** (FIXED, must not be rescaled — these are the SAME constants the
rule baseline uses, so that differences between rule and LLM are only about
which markings each encoder recognises, not the numerical ratios):

- `grave` 0.25, `largo` 0.27, `lento` 0.30, `larghetto` 0.37
- `adagio` 0.41, `adagietto` 0.50
- `andante` 0.67, `andantino` 0.74
- `moderato` 1.00, `allegretto` 1.28
- `allegro` 1.82, `allegro moderato` 1.57, `allegro non troppo` 1.65
- `vivace` 2.23, `vivo` 2.23
- `presto` 3.00, `prestissimo` 3.67
- `a tempo` 1.00, `tempo primo` 1.00, `tempo i` 1.00
- `meno mosso` 0.74 (relative to prevailing), `più mosso` 1.35 (relative)
- `agitato` 1.49, `con moto` 1.22
- `sostenuto` 0.82, `tranquillo` 0.74
- `appassionato` 1.35, `animato` 1.35
- `maestoso` 0.67, `cantabile` 0.90

**Modifier conventions**:
- `ritenuto`, `rall.`, `riten.` apply 0.75× to the prevailing γ (no ramp —
  jump down at the marking, hold until the next tempo marking).
- `accelerando`, `accel.`, `stringendo` apply 1.35× to the prevailing γ (jump up).
- `poco` prefix weakens magnitude ×0.5 (blend toward 1.0).
- `sempre più` / `molto` strengthen ×1.3.
- `fermata` → γ=0.35 for ONE beat only; next beat returns to the prior γ.
- Unknown text → γ=1.0; annotate `"reasoning": "unknown marking, defaulted to 1.0"`.

**Section construction rules** (PWC):
1. Iterate markings in ascending beat order.
2. Between consecutive markings `M_i` (beat $b_i$, γ $g_i$) and `M_{i+1}`
   (beat $b_{i+1}$), emit exactly one section `[b_i, b_{i+1}, g_i, g_i]`.
3. Before the first marking: section `[0, b_0, 1.0, 1.0]` unless marking at
   beat 0.
4. After the last marking: section `[b_last, score_beat_max, g_last, g_last]`.
5. For fermatas: emit `[b, b+1, 0.35, 0.35]` followed by a restoration section
   `[b+1, next_boundary, prev_γ, prev_γ]`.

**Determinism**: temperature 0, same input → identical output (modulo
whitespace). Do not introduce randomness, do not second-guess the lookup.

---

## Input data block

```json
{{INPUT_JSON}}
```

Produce ONLY the output JSON. No commentary, no Markdown fencing.
