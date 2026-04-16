"""
Train Score Flower Model
=========================
Train conditional flow matching model on ASAP tempo trajectories.
"""
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))
from score_flower_model import (
    TempoTrajectoryDataset,
    TempoTransformer,
    FlowMatchingTempoModel,
    flow_matching_loss,
    generate_trajectory,
)

DATA_PATH = 'output/tempo_trajectories.pkl'
SAVE_DIR = 'output/score_flower'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
MAX_LEN = 512
CONTEXT_BEATS = 20
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 100


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Data
    dataset = TempoTrajectoryDataset(DATA_PATH, max_len=MAX_LEN,
                                     context_beats=CONTEXT_BEATS)
    print(f'Dataset: {len(dataset)} samples, {dataset.n_pieces} pieces')

    # Split
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    print(f'Train: {n_train}, Val: {n_val}')

    # Model: Flow Matching
    model = FlowMatchingTempoModel(
        n_pieces=dataset.n_pieces,
        max_len=MAX_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        context_beats=CONTEXT_BEATS,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,} ({n_params/1e6:.1f}M)')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            piece_idx = batch['piece_idx'].to(DEVICE)
            context = batch['context'].to(DEVICE)
            context_mask = batch['context_mask'].to(DEVICE)
            target = batch['target'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)

            loss = flow_matching_loss(model, piece_idx, context,
                                     context_mask, target, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                piece_idx = batch['piece_idx'].to(DEVICE)
                context = batch['context'].to(DEVICE)
                context_mask = batch['context_mask'].to(DEVICE)
                target = batch['target'].to(DEVICE)
                mask = batch['mask'].to(DEVICE)

                loss = flow_matching_loss(model, piece_idx, context,
                                         context_mask, target, mask)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:3d}: train={train_loss:.4f} val={val_loss:.4f} '
                  f'lr={scheduler.get_last_lr()[0]:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pt'))

    print(f'\nBest val loss: {best_val_loss:.4f}')

    # Generate sample trajectory
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pt')))
    model.eval()

    sample = dataset[0]
    piece_idx = torch.tensor([sample['piece_idx']], device=DEVICE)
    context = sample['context'].unsqueeze(0).to(DEVICE)
    context_mask = sample['context_mask'].unsqueeze(0).to(DEVICE)
    target = sample['target']

    pred = generate_trajectory(model, piece_idx, context, context_mask, n_steps=20)
    pred = pred.cpu().numpy()
    target = target.numpy()

    # Compute prediction error (on non-context beats)
    n_actual = int(sample['mask'].sum())
    pred_beats = pred[CONTEXT_BEATS:n_actual]
    target_beats = target[CONTEXT_BEATS:n_actual]
    mae = np.mean(np.abs(pred_beats - target_beats))
    corr = np.corrcoef(pred_beats, target_beats)[0, 1] if len(pred_beats) > 1 else 0

    print(f'\nSample prediction:')
    print(f'  Piece: {dataset.samples[0]["piece_idx"]}')
    print(f'  Beats: {n_actual} (context: {CONTEXT_BEATS}, predict: {n_actual-CONTEXT_BEATS})')
    print(f'  MAE: {mae:.4f}')
    print(f'  Correlation: {corr:.4f}')
    print(f'  Target first 10 (after context): {np.round(target_beats[:10], 3)}')
    print(f'  Pred   first 10 (after context): {np.round(pred_beats[:10], 3)}')

    # Save dataset info for later evaluation
    info = {
        'n_pieces': dataset.n_pieces,
        'piece_to_idx': dataset.piece_to_idx,
        'context_beats': CONTEXT_BEATS,
        'max_len': MAX_LEN,
    }
    with open(os.path.join(SAVE_DIR, 'info.pkl'), 'wb') as f:
        import pickle
        pickle.dump(info, f)


if __name__ == '__main__':
    train()
