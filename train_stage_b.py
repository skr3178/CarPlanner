"""
CarPlanner baseline — IL training (Stage B).

Loss: L_IL = L_CE + L_SideTask + L_generator  (equal weights, Section 3.4)
  L_CE        = CrossEntropy(mode_logits, c*)   Eq (6)
  L_SideTask  = L1(side_traj, gt_traj)          Eq (7)
  L_generator = L1(pred_traj, gt_traj)          Eq (11)

Optimizer: AdamW, lr=1e-4, ReduceLROnPlateau schedule (Section 4.1).
"""

import os
import sys
import argparse
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as cfg
from data_loader import make_dataloader
from model import CarPlanner


# ── Loss functions ─────────────────────────────────────────────────────────────

def compute_il_loss(mode_logits, side_traj, pred_traj, gt_traj, mode_label):
    """
    IL total loss (Section 3.4, equal weights).
    Returns scalar total and dict of components for logging.
    """
    # Eq (6): mode selector CE loss
    L_CE = F.cross_entropy(mode_logits, mode_label)

    # Eq (7): side task L1 — (1/T) Σ_t ||s_t - s_t^gt||_1  (sum over spatial, mean over T)
    if cfg.SELECTOR_SIDE_TASK:
        L_side = (side_traj - gt_traj).abs().sum(dim=-1).mean()
    else:
        L_side = torch.tensor(0.0, device=mode_logits.device)

    # Eq (11): generator L1 — (1/T) Σ_t ||s_t - s_t^gt||_1
    L_gen = (pred_traj - gt_traj).abs().sum(dim=-1).mean()

    L_total = L_CE + L_side + L_gen

    return L_total, {
        'L_CE': L_CE.item(),
        'L_side': L_side.item(),
        'L_gen': L_gen.item(),
        'L_total': L_total.item(),
    }


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")
    print(f"[Train] Split: {args.split}, Epochs: {args.epochs}, "
          f"Batch: {args.batch_size}")

    # Data
    loader = make_dataloader(
        args.split,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        max_per_file=args.max_per_file,
    )
    print(f"[Train] Dataset size: {len(loader.dataset)}, "
          f"Batches/epoch: {len(loader)}")

    # Model
    model = CarPlanner().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Parameters: {n_params:,}")

    # Load frozen transition model from Stage A if provided
    if args.transition_ckpt:
        model.load_transition_model(args.transition_ckpt, freeze=True)
        print(f"[Train] Loaded frozen transition model from: {args.transition_ckpt}")

    # Optimizer & scheduler (Section 4.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=cfg.LR_PATIENCE,
        factor=cfg.LR_FACTOR,
    )

    # Resume checkpoint if provided
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"[Train] Resumed from epoch {start_epoch}")

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {'L_CE': 0., 'L_side': 0., 'L_gen': 0., 'L_total': 0.}
        t0 = time.time()

        for batch_idx, batch in enumerate(loader):
            agents_now   = batch['agents_now'].to(device)            # (B, N, 4)
            agents_mask  = batch['agents_history_mask'].to(device)  # (B, N)
            agents_seq   = batch['agents_seq'].to(device)           # (B, T, N, 4)
            gt_traj      = batch['gt_trajectory'].to(device)        # (B, T, 3)
            mode_label   = batch['mode_label'].to(device)           # (B,)
            map_lanes    = batch['map_lanes'].to(device)            # (B, N_LANES, N_PTS, 3)
            map_lanes_mask = batch['map_lanes_mask'].to(device)     # (B, N_LANES)

            # Forward (Algorithm 1, Stage 2 IL branch)
            mode_logits, side_traj, pred_traj = model.forward_train(
                agents_now, agents_mask, agents_seq, gt_traj, mode_label,
                map_lanes=map_lanes, map_lanes_mask=map_lanes_mask
            )

            # Loss (L_IL = L_CE + L_SideTask + L_generator)
            L_total, loss_dict = compute_il_loss(
                mode_logits, side_traj, pred_traj, gt_traj, mode_label
            )

            # Backward
            optimizer.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k]

            if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}  "
                      f"[{batch_idx+1}/{len(loader)}]  "
                      f"L_total={loss_dict['L_total']:.4f}  "
                      f"L_CE={loss_dict['L_CE']:.4f}  "
                      f"L_gen={loss_dict['L_gen']:.4f}")

        # Epoch summary
        n = len(loader)
        avg = {k: v / n for k, v in epoch_losses.items()}
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"L_total={avg['L_total']:.4f}  "
              f"L_CE={avg['L_CE']:.4f}  "
              f"L_side={avg['L_side']:.4f}  "
              f"L_gen={avg['L_gen']:.4f}  "
              f"({elapsed:.1f}s)")

        scheduler.step(avg['L_total'])

        # Save checkpoint
        is_best = avg['L_total'] < best_loss
        best_loss = min(avg['L_total'], best_loss)

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg,
        }
        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"epoch_{epoch+1:03d}.pt")
        torch.save(ckpt, ckpt_path)
        if is_best:
            best_path = os.path.join(cfg.CHECKPOINT_DIR, "best.pt")
            torch.save(ckpt, best_path)
            print(f"  ★ New best saved: {best_path}")

    print(f"\n[Train] Done. Best L_total={best_loss:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CarPlanner IL training")
    p.add_argument('--split', default='mini',
                   choices=['mini', 'train_boston'])
    p.add_argument('--epochs', type=int, default=cfg.EPOCHS)
    p.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE)
    p.add_argument('--num_workers', type=int, default=cfg.NUM_WORKERS)
    p.add_argument('--max_per_file', type=int, default=None,
                   help='Cap on samples per DB file (default: from config)')
    p.add_argument('--resume', default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--transition_ckpt', default=None,
                   help='Path to Stage A checkpoint (loads frozen transition model)')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
