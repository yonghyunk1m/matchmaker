# SCORPION LLM Intent — Score-Only Prompt (v1)

**Version**: v1 (2026-04-15)
**Purpose**: Convert a MusicXML's printed Italian tempo/character markings into a
per-beat tempo-ratio trajectory $\gamma(b) \in (0, 5]$ that represents the
performer's *expected* intent before performance begins.

**Input to the LLM** (filled in at call time):
- `PIECE_ID`: piece identifier (e.g. `chopin_ballade1`)
- `SCORE_BEAT_MAX`: total beat count of the score (e.g. 1452)
- `MARKINGS`: ordered list of `{beat, text, kind}` extracted by partitura from
  the MusicXML. `kind ∈ {tempo, accel, rall, fermata}`.

**Output contract** (enforced JSON schema — any deviation is a failure):
```json
{
  "piece_id": "<string>",
  "score_beat_max": <float>,
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
    [<start_beat>, <end_beat>, <gamma_start>, <gamma_end>]
  ]
}
```

**Conventions** (fixed before any ASAP evaluation; `gamma=1` means the XML-encoded
score tempo, `gamma<1` means slower, `gamma>1` means faster):

- `grave` 0.25, `largo` 0.27, `lento` 0.30, `larghetto` 0.37
- `adagio` 0.41, `adagietto` 0.50
- `andante` 0.67, `andantino` 0.74
- `moderato` 1.00, `allegretto` 1.28
- `allegro` 1.82, `allegro moderato` 1.57, `allegro non troppo` 1.65
- `vivace` 2.23, `vivo` 2.23
- `presto` 3.00, `prestissimo` 3.67
- `a tempo` 1.00, `tempo primo` 1.00, `tempo i` 1.00
- `meno mosso` 0.74, `più mosso` 1.35
- `agitato` 1.49, `con moto` 1.22
- `sostenuto` 0.82, `tranquillo` 0.74
- `appassionato` 1.35, `animato` 1.35
- `maestoso` 0.67, `cantabile` 0.90
- `ritenuto`, `rall.`, `riten.` apply a 25% slowdown relative to the prevailing
  ratio (multiply by 0.75).
- `accelerando`, `accel.`, `stringendo` apply a 35% speedup (multiply by 1.35).
- `poco` prefix weakens magnitude ×0.5; `sempre più` / `molto` strengthen ×1.3.
- `fermata` is a local hold (set `gamma=0.35` for the immediate beat only).
- Unknown markings: reason from context; default `gamma=1` unless intent is clear.

**Rules for the `sections` array**:

1. Section boundaries **must** align with marking positions (within 0.5 beats).
2. Between two tempo markings, γ is **piecewise-linear**: start = γ at the first
   marking, end = γ at the next marking. If both markings are the same γ (no
   ramp in between), emit a flat section (`gamma_start == gamma_end`).
3. A `rall./ritenuto` preceding a `tempo` marking generates a **linear ramp from
   the previous tempo γ to 0.75× that γ** over the last 4 beats before the
   tempo marking, then jumps back to the new tempo.
4. An `accelerando` preceding a `tempo` marking generates a **linear ramp from
   the previous tempo γ to 1.35× that γ** over the last 4 beats.
5. Fermatas contribute a single-beat dip to γ=0.35 that does NOT propagate.
6. `sections` must fully cover `[0, score_beat_max]` with no gaps.

**Reasoning requirement**: for every marking in `markings_annotated`, output a
one-sentence `reasoning` explaining why that γ was chosen. This field is for
audit, not for downstream use.

**Determinism requirement**: calling this prompt with the same input at
`temperature=0` must yield the same output JSON modulo whitespace. The lookup
constants above are FIXED — do not rescale them even if they seem aggressive.

---

## Input data block (filled at call time)

```json
{{INPUT_JSON}}
```

Produce ONLY the output JSON. No commentary, no Markdown fencing.
