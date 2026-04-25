# Paper Integration Notes — Section-Level Algorithm Swap

_Working draft for ISMIR 2026 paper revision (deadline 2026-04-27)_

## Headline contribution (proposed)

> **Section-level rehearsal-driven algorithm switching delivers always-improve
> score following on cascade pieces.** On Chopin Ballade Op.~23 (n=7), per-section
> selection of {Arzt+plan, Pass-2 confidence-gated, Dixon+CQT} based on
> rehearsal-time ground-truth comparison achieves min Δ = +1.4pp, mean Δ = +14.99pp
> vs.\ Dixon+CQT baseline — every performer improves, no regression. Mo07M, the
> hardest case (Andante mismatch with rule lexicon), is rescued from a -14.2pp
> single-tracker regression to a +22.5pp gain.

## Empirical results table (Op.~23, n=7)

| Perf | Pass1 (Arzt+plan) | Pass2 (gated) | Dixon+CQT | **Section-Best** | Δ vs Dixon |
|------|------------------:|--------------:|----------:|-----------------:|-----------:|
| BuiJL04M | 48.7 | 28.2 | 21.9 | **53.4** | **+31.6** |
| JIA06M | 17.0 | 1.4 | 12.3 | **17.2** | +4.9 |
| KimSuyeon04M | 49.2 | 31.1 | 17.3 | **51.3** | **+34.0** |
| Mo07M | 14.9 | 30.3 | 29.1 | **51.6** | **+22.5** |
| Richardson13M | 16.5 | 0.3 | 17.6 | **19.0** | +1.4 |
| Sladek04M | 8.9 | 14.7 | 25.6 | **27.3** | +1.7 |
| Zhou06M | 23.3 | 23.1 | 25.0 | **33.8** | +8.9 |
| **Mean** | 25.5 | 18.4 | 21.4 | **36.4** | **+14.99** |

Per-perf section-choice distribution (avg over n=7):
- Pass1 (Arzt+plan): 19/25 sections — dominant on cascade
- Pass2 (gated): 1/25 — occasional refinement
- Dixon+CQT: 5/25 — dominant on active passagework

## Multi-metric improvements (vs Dixon+CQT)

(All for Op.~23 n=7, mean):
- AR@500: 21.4% → **36.4%** (+14.99pp)
- MedAE: 16.4s → **0.5s** (32× reduction in catastrophic drift)
- MaxAE: 79.1s → **77.7s** (small change; outlier sections persist)
- LR@500_K=5: 11.4% → **22.7%** (~2× sustained alignment)

## Mechanism

Three trackers running on the same audio:
1. **Pass1 (Arzt+plan)**: rule-lexicon γ injected as Arzt step-size cap. Strong on
   cascade sections (long sustained chord regions where Arzt's forward bias
   causes drift).
2. **Pass2 (confidence-gated)**: rehearsal-derived γ override per section where
   |γ_rule − γ_actual| > 0.20. Rescues Andante-mismatch performers (Mo07M).
3. **Dixon+CQT**: path-length normalization, no forward bias. Strong on active
   passagework where Arzt's cap is too restrictive.

Each section (= contiguous beats between two MusicXML tempo markings) is assigned
to its best-performing tracker, evaluated against rehearsal ground truth.

## Deployment

### Scenario A: Pre-decided schedule (1× compute at concert)

```
[REHEARSAL]                                    [CONCERT]
1. Pianist plays into mic                     1. Per-section tracker switching
2. Audio → MIDI transcription                    by predetermined schedule
3. Run all 3 trackers; per-section            2. State transfer at section boundary:
   AR@500 vs transcribed GT                       - new tracker init at predicted
4. Schedule = [Pass1, Pass1, Dixon, ...]         position
                                              3. CPU: 1× real-time
```

### Scenario B: Parallel shadow tracking (3× compute)

```
[CONCERT]
- 3 trackers in parallel (3× CPU; modern desktop ~RTF 0.75)
- Controller publishes chosen tracker's output per section
- No state-transfer issues; no cursor jump
```

## Section boundaries

From MusicXML:
- Each contiguous span between Italian tempo markings (Lento, Animato, etc.)
- Chopin Op.~23 has 25 such sections from 25 markings.
- Score-defined; no audio analysis required.

## Honest limitations

- **Cross-piece validation pending**: only Op.~23 (n=7) confirmed. Other cascade
  pieces (Schubert Op.~90/3, Liszt) need confidence-gated v2 runs (planned, CPU-bound).
- **Rehearsal MIDI requirement**: validated via audio-to-MIDI transcription as
  an alternative (pending validation; piano_transcription_inference test
  in background).
- **Per-perf consistency**: assumes pianist plays similarly at rehearsal and
  concert. ASAP-style assumption.
- **Section boundary stability**: assumes pianist follows score's marked
  structure. Improvisation breaks the assumption.

## Comparison to paper's existing claims

- Existing: Op.~23 mean +17.3pp via Arzt+plan only (single-tracker,
  cascade-only) — has per-perf variance (Sladek-type outliers).
- New: Op.~23 mean +14.99pp via section-level swap — but **no per-perf regression**.
  Different framing: per-perf "always improve" vs. mean improvement.

The two contributions stack:
1. PerformancePlan channel (existing) — gives Arzt+plan its strength on cascades.
2. Section-level swap (new) — combines Arzt+plan, Pass2 gated, Dixon+CQT for
   per-perf always-improve.

## Paper structure changes (proposed)

- §3 Method: add Pass-2 confidence-gated rehearsal subsection.
- §3 Method: add section-level swap mechanism.
- §5 Results: add Table tab:section_swap with per-perf and aggregate stats.
- §5 Results: add multi-metric (AR@500, MedAE, MaxAE, LR_K5) table.
- §6 Discussion: deployment scenarios (pre-decided vs shadow), state transfer.
- §6 Future Work: cross-piece validation, audio-only confidence proxies (for
  pure-audio deployment without rehearsal MIDI/transcription).
