"""
CarPlanner — Stage A: Transition Model Pre-training.

Trains P_τ(s^{1:N}_{1:T} | s_0) with masked L1 loss (Eq 5).
The transition model predicts future agent states given current agent states.
Pre-trained here, then frozen during Stages B and C.

Usage:
    python train_stage_a.py --split mini --epochs 5 --batch_size 4
    python train_stage_a.py --split mini --epochs 10 --resume checkpoints/stage_a_best.pt
"""

import os
import argparse
import random
import time

import torch

import config as cfg
from data_loader import make_dataloader
from model import TransitionModel


# ── Loss function (Eq 5) ──────────────────────────────────────────────────────

def compute_transition_loss(pred, gt, mask):
    """
    Masked L1 loss for agent future prediction (Eq 5).

    Args:
        pred: (B, T, N, Da) — predicted agent futures
        gt:   (B, T, N, Da) — ground-truth agent futures (agents_seq)
        mask: (B, N)        — agent validity mask (1=valid, 0=padding)

    Returns:
        Scalar loss: mean L1 over valid (B, T, N) entries.
    """
    diff = (pred - gt).abs().sum(dim=-1)        # (B, T, N) — L1 per agent per step
    mask_t = mask.unsqueeze(1).expand_as(diff)  # (B, T, N) — broadcast mask over time
    return (diff * mask_t).sum() / (mask_t.sum() + 1e-6)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Stage A] Transition Model Pre-training")
    print(f"[Stage A] Device: {device}")
    print(f"[Stage A] Split: {args.split}, Epochs: {args.epochs}, "
          f"Batch: {args.batch_size}")

    # Data — same dataloader as Stage B, we only use agent-related fields
    loader = make_dataloader(
        args.split,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        max_per_file=args.max_per_file,
    )
    print(f"[Stage A] Dataset size: {len(loader.dataset)}, "
          f"Batches/epoch: {len(loader)}")

    # Only instantiate TransitionModel (not full CarPlanner)
    model = TransitionModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Stage A] TransitionModel parameters: {n_params:,}")

    # Optimizer & scheduler (same hyperparams as Stage B, Section 4.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=cfg.LR_PATIENCE,
        factor=cfg.LR_FACTOR,
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"[Stage A] Resumed from epoch {start_epoch}")

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # Preload all data to GPU once — avoids repeated CPU map queries
    # Cache is saved to disk so subsequent runs skip the expensive dataloader
    cache_path = os.path.join(
        cfg.CHECKPOINT_DIR, f"stage_a_cache_{args.split}_bs{args.batch_size}.pt"
    )
    if args.preload and args.split == 'train_boston':
        print(f"[Stage A] WARNING: --preload with train_boston may OOM GPU memory "
              f"(~10GB+). Consider running without --preload.")
    if args.preload:
        if os.path.isfile(cache_path):
            print(f"[Stage A] Loading cached data from {cache_path}...")
            t0 = time.time()
            cached = torch.load(cache_path, map_location=device)
            print(f"[Stage A] Loaded cache in {time.time() - t0:.1f}s "
                  f"({len(cached)} batches)")
        else:
            print(f"[Stage A] Preloading {len(loader)} batches into GPU memory...")
            cached = []
            t_preload = time.time()
            for batch in loader:
                cached.append({
                    'agents_now':  batch['agents_now'].to(device),
                    'agents_mask': batch['agents_history_mask'].to(device),
                    'agents_seq':  batch['agents_seq'].to(device),
                })
            elapsed = time.time() - t_preload
            print(f"[Stage A] Preloaded in {elapsed:.1f}s — saving cache...")
            torch.save(cached, cache_path)
            print(f"[Stage A] Cache saved to {cache_path}")
    else:
        cached = None

    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        batch_iter = random.sample(cached, len(cached)) if cached else loader
        for batch_idx, batch in enumerate(batch_iter):
            if cached:
                agents_now  = batch['agents_now']
                agents_mask = batch['agents_mask']
                agents_seq  = batch['agents_seq']
            else:
                agents_now  = batch['agents_now'].to(device)
                agents_mask = batch['agents_history_mask'].to(device)
                agents_seq  = batch['agents_seq'].to(device)

            # Forward: predict agent futures from current state
            agents_pred = model(agents_now, agents_mask)            # (B, T, N, Da)

            # Loss: masked L1 (Eq 5)
            loss = compute_transition_loss(agents_pred, agents_seq, agents_mask)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            n_batches = len(cached) if cached else len(loader)
            if (batch_idx + 1) % max(1, n_batches // 5) == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}  "
                      f"[{batch_idx+1}/{len(loader)}]  "
                      f"L_tm={loss.item():.4f}")

        # Epoch summary
        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"L_tm={avg_loss:.4f}  "
              f"({elapsed:.1f}s)")

        scheduler.step(avg_loss)

        # Save checkpoint
        is_best = avg_loss < best_loss
        best_loss = min(avg_loss, best_loss)

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }
        ckpt_path = os.path.join(
            cfg.CHECKPOINT_DIR, f"stage_a_epoch_{epoch+1:03d}.pt"
        )
        torch.save(ckpt, ckpt_path)
        if is_best:
            best_path = os.path.join(cfg.CHECKPOINT_DIR, "stage_a_best.pt")
            torch.save(ckpt, best_path)
            print(f"  * New best saved: {best_path}")

    print(f"\n[Stage A] Done. Best L_tm={best_loss:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CarPlanner Stage A: Transition Model Pre-training"
    )
    p.add_argument('--split', default='mini',
                   choices=['mini', 'train_boston'])
    p.add_argument('--epochs', type=int, default=cfg.EPOCHS)
    p.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE)
    p.add_argument('--num_workers', type=int, default=cfg.NUM_WORKERS)
    p.add_argument('--max_per_file', type=int, default=None,
                   help='Cap on samples per DB file (default: from config)')
    p.add_argument('--preload', action='store_true',
                   help='Preload all data into GPU memory before training')
    p.add_argument('--resume', default=None,
                   help='Path to Stage A checkpoint to resume from')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
