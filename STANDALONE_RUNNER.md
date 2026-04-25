# Standalone Runner — Cross-piece Confidence-Gated v2 + Section-Swap

This package allows a **separate machine's Claude Code** to run cross-piece
experiments in parallel with the main server, then sync results back.

## What this does

For each piece in `PIECES_TO_RUN`:
1. Generates **Pass-1** (Arzt + rule plan) trajectory per performer
2. Generates **Pass-2** (confidence-gated rehearsal) trajectory per performer
3. Generates **Dixon+CQT** baseline trajectory per performer
4. Outputs CSVs to `results/confidence_gated_v2/trajectories/`
5. Builds a per-piece summary table

The downstream **section-swap analysis** (`analyze_section_swap.py`) consumes
these CSVs to compute per-section algorithm-best selection.

## Setup on remote machine

```bash
# 1. Clone the repo (matchmaker fork)
git clone -b develop https://github.com/yonghyunk1m/matchmaker.git
cd matchmaker

# 2. Python env (Python ≥ 3.10)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements_standalone.txt

# 3. ASAP dataset
#    Required structure:
#      datasets/asap-dataset/<Composer>/<piece>/<piece_subdir>/
#         ├─ xml_score.musicxml
#         └─ <PerformerID>.wav  + <PerformerID>.mid   (per ASAP convention)
#    If you have ASAP locally, symlink or copy into datasets/asap-dataset/
#    Otherwise: clone ASAP from https://github.com/cpjku/asap-dataset
```

## Required files (already in repo)

- `run_full_experiment.py` — defines PIECES list, run_single, load_gt
- `run_directional_plan_pilot.py` — metrics_from_err, helpers
- `run_directional_dixon_pilot.py` — run_dixon
- `run_scorpion_rule_intent.py` — extract_rule_gamma, TEMPO_TABLE
- `run_confidence_gated_v2.py` — pass1/pass2 mechanism
- `run_confidence_gated_all_cascade.py` — entry point for all cascade pieces
- `analyze_section_swap.py` — post-hoc analysis (run AFTER trajectories collected)
- `cached_pipeline.py` — optional caching layer (can speed up if used)
- `matchmaker/` — modified Matchmaker library (Arzt step-size cap, etc.)

## How to run

### Entry point: assign distinct piece subset per machine

Open `run_confidence_gated_all_cascade.py` and **at the top of `main()`**, restrict
`PIECES_TO_RUN` to the subset assigned to this machine. Example:

```python
# In main(), filter PIECES list:
ASSIGNED = {"liszt_sonata", "schubert_sonata_b_960_4", "scriabin_etude_op8_no12"}
for piece in PIECES:
    pid = piece["id"]
    if pid not in ASSIGNED:
        continue
    # ... existing loop body
```

This way:
- **Machine A (your laptop)** runs: chopin_ballade1, chopin_ballade_2, schubert_impromptu_3
- **Machine B (remote)** runs: liszt_sonata, scriabin_etudes, ravel_pieces, etc.

### Execute

```bash
# In the matchmaker root:
mkdir -p results/confidence_gated_v2/trajectories
python3 run_confidence_gated_all_cascade.py 2>&1 | tee /tmp/cgated_$(hostname).log
```

Each (piece, performer) takes 100-200 seconds for pass-1 + pass-2 + Dixon
baseline (~5 min per perf). Op.~23 has 7 perfs ≈ 35 min; Liszt sonata has 10
perfs ≈ 50 min, etc.

## What to send back

After the run finishes, sync these directories to me:

```bash
# Compress just the trajectory CSVs (small, ~20 KB each)
cd results
tar czf cgated_$(hostname).tar.gz \
    confidence_gated_v2/trajectories/ \
    confidence_gated_v2/all_cascade_pass2.csv

# Either:
#  (a) Upload to a shared bucket / GDrive
#  (b) Push to a branch of the matchmaker repo:
git checkout -b cgated-from-$(hostname)
git add -f results/confidence_gated_v2/
git commit -m "cgated trajectories from $(hostname)"
git push origin cgated-from-$(hostname)

#  (c) scp to my machine:
scp cgated_$(hostname).tar.gz ykim3098@<my_server>:/home/ykim3098/matchmaker/results/
```

## Verifying correctness

After the run, before sending:
```bash
# Quick sanity check: every perf should have 2 CSVs (pass1 + pass2)
ls results/confidence_gated_v2/trajectories/ | grep _pass1_v2.csv | wc -l
ls results/confidence_gated_v2/trajectories/ | grep _pass2_v2.csv | wc -l
# These two numbers should be equal (per perf, both files exist).

# Each CSV should have ~hundreds of rows
wc -l results/confidence_gated_v2/trajectories/*_pass1_v2.csv | head
```

## Once I receive the data

I'll merge into the master `results/` directory and re-run:
```bash
python3 analyze_section_swap.py    # ← cross-piece section swap report
python3 build_full_metrics_table.py # ← multi-metric markdown
```

## Failure modes / gotchas

1. **MusicXML missing for a piece**: script logs `SKIP {pid}: no xml` and moves on.
   Don't worry — just send back what worked.

2. **`run_single` raises**: check whether ASAP dataset path matches what
   `run_full_experiment.py:DATASET_ROOT` expects (default
   `/path/to/matchmaker/datasets/asap-dataset/`). Edit if your layout differs.

3. **Slow render**: first run per piece does FluidSynth rendering (slow);
   subsequent runs reuse `mm.score_audio` per perf only within the same
   `run_single` call. If integrating `cached_pipeline.py`, render is cached
   on disk under `cache/pipeline/`.

4. **OOM**: each perf uses ~2 GB during Matchmaker DTW. With 4-core CPU and
   sequential perfs, peak memory is one perf's footprint.

## Communication channel

Once running, please post status updates / completion to the GitHub branch's
PR description, or send the tarball directly. I'll periodically check the
branch and pull when ready.
