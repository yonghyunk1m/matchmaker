# Score Flower: Flow-Based Tempo Prediction for Anticipatory Score Following

## Overview
Predict a performer's tempo trajectory from the first few beats of a live performance using conditional flow matching. The predicted trajectory is injected into a score follower to enable anticipatory tracking.

## Quick Start

### 1. Prepare Data
```bash
python prepare_tempo_data.py
# Output: output/tempo_trajectories.pkl (1,067 trajectories from ASAP)
```

### 2. Train Model
```bash
python train_score_flower.py
# Output: output/score_flower/best_model.pt
# ~15 min on CPU, ~1 min on GPU
```

### 3. Evaluate
```bash
python eval_score_flower.py  # TODO
```

## Files

| File | Description |
|------|-------------|
| `prepare_tempo_data.py` | Extract per-beat tempo profiles from ASAP dataset |
| `score_flower_model.py` | Model definitions (TempoTransformer, FlowMatchingTempoModel) |
| `train_score_flower.py` | Training script (100 epochs, flow matching loss) |
| `output/tempo_trajectories.pkl` | 1,067 tempo trajectories (pre-extracted) |
| `output/score_flower/best_model.pt` | Trained model checkpoint |

## Architecture

- **Input**: Piece ID (one-hot) + first 20 beats of observed tempo ratios
- **Output**: Per-beat tempo ratios for entire piece (up to 512 beats)
- **Model**: Transformer encoder (4 layers, 128 dim, 4 heads, 0.9M params)
- **Training**: Conditional flow matching (linear interpolation path)
- **Inference**: 5-20 Euler steps from noise → trajectory

## Data

- **Source**: ASAP dataset (1,067 piano performances, 298 pieces)
- **Format**: Per-beat tempo ratios (local tempo / median tempo)
- **Split**: Random 90/10 (same piece can appear in train/test — this is the realistic scenario where the model knows the piece but not the performer)

## Current Status

- MAE: 0.093 (9.3% tempo ratio error)
- Correlation: 0.026 (poor — model predicts near-average, doesn't capture variation patterns)
- **Needs**: More training (GPU), larger model, or architecture changes

## Dependencies

```
torch
numpy
mido
pickle
```

## GPU Training

For faster training with hyperparameter search:
```bash
# On GPU server:
CUDA_VISIBLE_DEVICES=0 python train_score_flower.py
```

Suggested improvements:
- Increase epochs: 100 → 1000
- Increase model: d_model=128 → 256, n_layers=4 → 8
- Add score features (not just piece ID): note density, key, time signature
- Try different context sizes: 10, 20, 50 beats
- Ablation: with vs without context (to verify model uses performer info)
