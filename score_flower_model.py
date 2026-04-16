"""
Score Flower: Flow-Based Tempo Trajectory Prediction
=====================================================
Given the first N beats of a performance, predict the entire
tempo trajectory using conditional flow matching.

Architecture:
  - Condition: piece embedding + observed tempo (first N beats)
  - Target: full tempo trajectory (per-beat tempo ratios)
  - Flow: linear interpolation from noise to target, conditioned
  - Model: small 1D U-Net or Transformer

Simplified version (v1): Conditional MLP/Transformer
  - No flow matching yet, direct regression
  - Input: [piece_onehot, observed_tempo_padded]
  - Output: predicted tempo trajectory
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Dataset
# =============================================================================

class TempoTrajectoryDataset(Dataset):
    """Dataset of tempo trajectories from ASAP performances."""

    def __init__(self, data_path, max_len=512, context_beats=20):
        """
        Args:
            data_path: path to tempo_trajectories.pkl
            max_len: max sequence length (pad/truncate)
            context_beats: number of observed beats as context
        """
        with open(data_path, 'rb') as f:
            raw = pickle.load(f)

        self.context_beats = context_beats
        self.max_len = max_len

        # Build piece vocabulary
        piece_ids = sorted(set(d['piece_id'] for d in raw))
        self.piece_to_idx = {p: i for i, p in enumerate(piece_ids)}
        self.n_pieces = len(piece_ids)

        # Process trajectories
        self.samples = []
        for d in raw:
            tr = d['tempo_ratios'].astype(np.float32)
            if len(tr) < context_beats + 10:
                continue  # too short

            # Pad or truncate to max_len
            if len(tr) > max_len:
                tr = tr[:max_len]
            pad_len = max_len - len(tr)
            mask = np.ones(len(tr), dtype=np.float32)
            if pad_len > 0:
                tr = np.pad(tr, (0, pad_len), constant_values=1.0)
                mask = np.pad(mask, (0, pad_len), constant_values=0.0)

            self.samples.append({
                'piece_idx': self.piece_to_idx[d['piece_id']],
                'trajectory': tr,
                'mask': mask,
                'median_tempo': d['median_tempo'],
                'n_beats': min(d['n_beats'], max_len),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        traj = torch.tensor(s['trajectory'])
        mask = torch.tensor(s['mask'])
        piece_idx = s['piece_idx']

        # Context: first N beats (observed)
        context = traj[:self.context_beats].clone()
        # Context mask: 1 for observed, 0 for to-predict
        context_mask = torch.zeros(self.max_len)
        context_mask[:self.context_beats] = 1.0

        return {
            'piece_idx': piece_idx,
            'context': context,
            'context_mask': context_mask,
            'target': traj,
            'mask': mask,
        }


# =============================================================================
# Model v1: Conditional Transformer for direct regression
# =============================================================================

class TempoTransformer(nn.Module):
    """Predict full tempo trajectory from partial observation + piece ID."""

    def __init__(self, n_pieces, max_len=512, d_model=128, n_heads=4,
                 n_layers=4, context_beats=20):
        super().__init__()
        self.max_len = max_len
        self.context_beats = context_beats
        self.d_model = d_model

        # Embeddings
        self.piece_emb = nn.Embedding(n_pieces, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.tempo_proj = nn.Linear(1, d_model)
        self.context_flag = nn.Embedding(2, d_model)  # 0=predict, 1=observed

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, piece_idx, context, context_mask, target=None):
        """
        Args:
            piece_idx: (B,) piece indices
            context: (B, context_beats) observed tempo ratios
            context_mask: (B, max_len) 1=observed, 0=predict
            target: (B, max_len) full trajectory (for teacher forcing)

        Returns:
            pred: (B, max_len) predicted tempo trajectory
        """
        B = piece_idx.shape[0]
        device = piece_idx.device

        # Build input sequence
        # For observed beats: use actual tempo
        # For predicted beats: use 1.0 (neutral) as input
        if target is not None:
            # Teacher forcing: use noisy target
            noise = torch.randn_like(target) * 0.1
            inp = target + noise
            inp = inp * context_mask + (1.0) * (1 - context_mask)
        else:
            inp = torch.ones(B, self.max_len, device=device)
            inp[:, :self.context_beats] = context

        # Embeddings
        positions = torch.arange(self.max_len, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_emb(positions)
        piece_emb = self.piece_emb(piece_idx).unsqueeze(1).expand(-1, self.max_len, -1)
        tempo_emb = self.tempo_proj(inp.unsqueeze(-1))
        ctx_flag = self.context_flag(context_mask.long())

        x = pos_emb + piece_emb + tempo_emb + ctx_flag

        # Causal mask: each position can attend to all previous + all observed
        # (not strictly causal — observed beats are always visible)
        x = self.transformer(x)

        pred = self.out_proj(x).squeeze(-1)
        # Blend: keep observed values, predict the rest
        pred = context_mask * inp + (1 - context_mask) * pred

        return pred


# =============================================================================
# Model v2: Conditional Flow Matching
# =============================================================================

class FlowMatchingTempoModel(nn.Module):
    """Flow matching for tempo trajectory generation.

    Instead of direct regression, learn a vector field that transforms
    noise into the target trajectory, conditioned on piece + context.
    """

    def __init__(self, n_pieces, max_len=512, d_model=128, n_heads=4,
                 n_layers=4, context_beats=20):
        super().__init__()
        self.max_len = max_len
        self.context_beats = context_beats

        # Same backbone as v1
        self.piece_emb = nn.Embedding(n_pieces, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.tempo_proj = nn.Linear(1, d_model)
        self.context_flag = nn.Embedding(2, d_model)
        self.time_emb = nn.Linear(1, d_model)  # flow time t ∈ [0,1]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, piece_idx, context, context_mask, x_t, t):
        """Predict velocity field v(x_t, t | piece, context).

        Args:
            piece_idx: (B,)
            context: (B, context_beats)
            context_mask: (B, max_len)
            x_t: (B, max_len) noisy trajectory at time t
            t: (B,) flow time in [0, 1]

        Returns:
            v: (B, max_len) predicted velocity
        """
        B = piece_idx.shape[0]
        device = piece_idx.device

        positions = torch.arange(self.max_len, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_emb(positions)
        piece_emb = self.piece_emb(piece_idx).unsqueeze(1).expand(-1, self.max_len, -1)
        tempo_emb = self.tempo_proj(x_t.unsqueeze(-1))
        ctx_flag = self.context_flag(context_mask.long())
        t_emb = self.time_emb(t.unsqueeze(-1)).unsqueeze(1).expand(-1, self.max_len, -1)

        x = pos_emb + piece_emb + tempo_emb + ctx_flag + t_emb
        x = self.transformer(x)
        v = self.out_proj(x).squeeze(-1)

        # Zero velocity for observed beats (they don't change)
        v = v * (1 - context_mask)

        return v


def flow_matching_loss(model, piece_idx, context, context_mask, target, mask):
    """Compute flow matching loss (conditional).

    Linear interpolation: x_t = (1-t) * x_0 + t * x_1
    where x_0 ~ N(1, 0.1) (noise around neutral tempo)
          x_1 = target trajectory
    Target velocity: v = x_1 - x_0
    """
    B = target.shape[0]
    device = target.device

    # Sample time t ~ U[0, 1]
    t = torch.rand(B, device=device)

    # Sample noise (centered around 1.0 = neutral tempo)
    x_0 = 1.0 + torch.randn_like(target) * 0.3

    # Interpolate
    t_expand = t.unsqueeze(-1)
    x_t = (1 - t_expand) * x_0 + t_expand * target

    # Force observed beats to stay at context values
    x_t[:, :model.context_beats] = context

    # Target velocity
    v_target = target - x_0

    # Predict velocity
    v_pred = model(piece_idx, context, context_mask, x_t, t)

    # Loss (only on non-padded, non-observed beats)
    loss_mask = mask * (1 - context_mask)
    loss = ((v_pred - v_target) ** 2 * loss_mask).sum() / loss_mask.sum().clamp(min=1)

    return loss


@torch.no_grad()
def generate_trajectory(model, piece_idx, context, context_mask, n_steps=10):
    """Generate tempo trajectory via ODE integration.

    Args:
        model: FlowMatchingTempoModel
        piece_idx: (1,)
        context: (1, context_beats)
        context_mask: (1, max_len)
        n_steps: number of Euler steps

    Returns:
        trajectory: (max_len,) predicted tempo ratios
    """
    device = piece_idx.device
    max_len = model.max_len

    # Start from noise
    x = 1.0 + torch.randn(1, max_len, device=device) * 0.3
    x[:, :model.context_beats] = context

    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.tensor([i * dt], device=device)
        v = model(piece_idx, context, context_mask, x, t)
        x = x + v * dt
        # Keep observed beats fixed
        x[:, :model.context_beats] = context

    return x.squeeze(0)
