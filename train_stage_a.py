"""
CarPlanner — Stage A: Transition Model Pre-training.

Trains P_tau(s^{1:N}_{1:T} | s_0) with masked L1 loss (Eq 5).
The transition model predicts future agent states given current agent states.
Pre-trained here, then frozen during Stages B and C.

Data pipeline:
  1. Extract once:   python extract_stage_a.py --split mini
  2. Train from cache: python train_stage_a.py --split mini --epochs 50 --batch_size 256

The cache file is a single .pt with all tensors — no SQLite, no map API at train time.
If no cache exists, falls back to the slow DataLoader (for testing/debugging only).

Usage:
    python train_stage_a.py --split mini --epochs 50 --batch_size 256
    python train_stage_a.py --split mini --epochs 50 --resume checkpoints/stage_a_best.pt
"""

import os
import sys
import argparse
import random
import time

# Flush stdout immediately so nohup log updates in real time
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import torch

import config as cfg
from data_loader import make_dataloader, make_cached_dataloader
from model import TransitionModel


# ── Loss function (Eq 5) ──────────────────────────────────────────────────────

def compute_transition_loss(pred, gt, mask):
    """
    Masked L1 loss for agent future prediction (Eq 5).

    Only computes loss over motion dims 0-7 (x, y, heading, vx, vy, box_w, box_l, box_h).
    Dims 8 (time_step) and 9 (category) are excluded — time_step is a deterministic
    ramp and category is a fixed label, neither is a prediction target.

    Args:
        pred: (B, T, N, Da) — predicted agent futures
        gt:   (B, T, N, Da) — ground-truth agent futures (agents_seq)
        mask: (B, N)        — agent validity mask (1=valid, 0=padding)

    Returns:
        Scalar loss: mean L1 over valid (B, T, N) entries on motion dims only.
    """
    diff = (pred[..., :8] - gt[..., :8]).abs().sum(dim=-1)  # (B, T, N) — L1 on motion dims
    mask_t = mask.unsqueeze(1).expand_as(diff)               # (B, T, N)
    return (diff * mask_t).sum() / (mask_t.sum() + 1e-6)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Stage A] Transition Model Pre-training")
    print(f"[Stage A] Device: {device}")
    print(f"[Stage A] Split: {args.split}, Epochs: {args.epochs}, "
          f"Batch: {args.batch_size}")

    # ── Data loading: cache-first, slow fallback ────────────────────────────
    cache_path = os.path.join(
        cfg.CHECKPOINT_DIR, f"stage_cache_{args.split}.pt"
    )
    use_cache = os.path.isfile(cache_path)
    pin_gpu = args.pin_gpu and device.type == 'cuda'

    if use_cache and pin_gpu:
        # GPU-pinned path: load entire cache to GPU once, zero transfers after
        print(f"[Stage A] Loading cache to GPU (pin_gpu): {cache_path}")
        t0 = time.time()
        data = torch.load(cache_path, map_location=device)
        gpu_cache = {
            'agents_history':      data['agents_history'],
            'agents_mask':         data['agents_mask'],
            'agents_seq':          data['agents_seq'],
            'map_lanes':           data['map_lanes'],
            'map_lanes_mask':      data['map_lanes_mask'],
            'map_polygons':        data.get('map_polygons', torch.zeros(data['n_samples'], cfg.N_POLYGONS, cfg.N_LANE_POINTS, cfg.D_POLYGON_POINT, device=device)),
            'map_polygons_mask':   data.get('map_polygons_mask', torch.zeros(data['n_samples'], cfg.N_POLYGONS, device=device)),
        }
        n_samples = data['n_samples']

        # Train/val split (90/10)
        n_val = max(1, int(n_samples * 0.1))
        n_train = n_samples - n_val
        perm = torch.randperm(n_samples, device=device)
        train_idx, val_idx = perm[:n_train], perm[n_train:]
        gpu_cache['train_idx'] = train_idx
        gpu_cache['val_idx'] = val_idx
        print(f"[Stage A] GPU cache: {n_samples} samples loaded in "
              f"{time.time()-t0:.1f}s, ~{torch.cuda.memory_allocated()/1e9:.1f}GB GPU used")
        print(f"[Stage A] Split: {n_train} train, {n_val} val")
        loader = None
        all_batches = None
        n_train_samples = n_train
        n_val_samples = n_val
    elif use_cache:
        print(f"[Stage A] Loading pre-extracted cache: {cache_path}")
        loader = make_cached_dataloader(
            cache_path,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,            # data already in memory, no workers needed
        )
        # For shuffling cached batches across epochs
        all_batches = list(loader)
        # Train/val split (90/10) on batches
        n_val_batches = max(1, int(len(all_batches) * 0.1))
        n_train_batches = len(all_batches) - n_val_batches
        print(f"[Stage A] Cache loaded: {len(loader.dataset)} samples, "
              f"{len(all_batches)} batches (train={n_train_batches}, val={n_val_batches})")
    else:
        print(f"[Stage A] WARNING: No cache found at {cache_path}")
        print(f"[Stage A] Falling back to slow DataLoader (SQLite per sample).")
        print(f"[Stage A] Run 'python extract_stage_a.py --split {args.split}' first.")
        loader = make_dataloader(
            args.split,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            max_per_file=args.max_per_file,
        )
        all_batches = None
        print(f"[Stage A] Dataset size: {len(loader.dataset)}, "
              f"Batches/epoch: {len(loader)}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = TransitionModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Stage A] TransitionModel parameters: {n_params:,}")

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

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        # GPU-pinned fast path: direct tensor indexing, zero transfers
        if pin_gpu:
            # Shuffle train indices only
            train_perm = torch.randperm(n_train_samples, device=device)
            n_batches_epoch = (n_train_samples + args.batch_size - 1) // args.batch_size

            for b in range(n_batches_epoch):
                idx = train_idx[train_perm[b * args.batch_size : (b + 1) * args.batch_size]]

                agents_history = gpu_cache['agents_history'][idx]
                agents_mask = gpu_cache['agents_mask'][idx]
                agents_seq = gpu_cache['agents_seq'][idx]
                map_lanes = gpu_cache['map_lanes'][idx]
                map_lanes_mask = gpu_cache['map_lanes_mask'][idx]
                map_polygons = gpu_cache['map_polygons'][idx]
                map_polygons_mask = gpu_cache['map_polygons_mask'][idx]

                agents_pred = model(agents_history, agents_mask, map_lanes, map_lanes_mask,
                                    map_polygons=map_polygons, map_polygons_mask=map_polygons_mask)
                loss = compute_transition_loss(agents_pred, agents_seq, agents_mask)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                if (b + 1) % max(1, n_batches_epoch // 5) == 0:
                    print(f"  Epoch {epoch+1}/{args.epochs}  "
                          f"[{b+1}/{n_batches_epoch}]  "
                          f"L_tm={loss.item():.4f}")

        # Non-pinned path: shuffle cached batches, or use DataLoader
        else:
            if use_cache:
                # Split into train/val batches
                shuffled = random.sample(all_batches, len(all_batches))
                epoch_batches = shuffled[:n_train_batches]
            else:
                epoch_batches = loader

            for batch_idx, batch in enumerate(epoch_batches):
                agents_history = batch['agents_history'].to(device)
                agents_mask = batch['agents_history_mask'].to(device)
                agents_seq = batch['agents_seq'].to(device)
                map_lanes = batch['map_lanes'].to(device) if 'map_lanes' in batch else None
                map_lanes_mask = batch['map_lanes_mask'].to(device) if 'map_lanes_mask' in batch else None
                map_polygons = batch['map_polygons'].to(device) if 'map_polygons' in batch else None
                map_polygons_mask = batch['map_polygons_mask'].to(device) if 'map_polygons_mask' in batch else None

                agents_pred = model(agents_history, agents_mask, map_lanes, map_lanes_mask,
                                    map_polygons=map_polygons, map_polygons_mask=map_polygons_mask)

                # Loss: masked L1 (Eq 5)
                loss = compute_transition_loss(agents_pred, agents_seq, agents_mask)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                if (batch_idx + 1) % max(1, len(epoch_batches) // 5) == 0:
                    print(f"  Epoch {epoch+1}/{args.epochs}  "
                          f"[{batch_idx+1}/{len(epoch_batches)}]  "
                          f"L_tm={loss.item():.4f}")

        avg_train_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        if pin_gpu:
            n_val_batches = (n_val_samples + args.batch_size - 1) // args.batch_size
            with torch.no_grad():
                for b in range(n_val_batches):
                    idx = val_idx[b * args.batch_size : (b + 1) * args.batch_size]
                    agents_history = gpu_cache['agents_history'][idx]
                    agents_mask = gpu_cache['agents_mask'][idx]
                    agents_seq = gpu_cache['agents_seq'][idx]
                    map_lanes = gpu_cache['map_lanes'][idx]
                    map_lanes_mask = gpu_cache['map_lanes_mask'][idx]
                    map_polygons = gpu_cache['map_polygons'][idx]
                    map_polygons_mask = gpu_cache['map_polygons_mask'][idx]

                    agents_pred = model(agents_history, agents_mask, map_lanes, map_lanes_mask,
                                        map_polygons=map_polygons, map_polygons_mask=map_polygons_mask)
                    loss = compute_transition_loss(agents_pred, agents_seq, agents_mask)
                    val_loss_sum += loss.item()
                    val_batches += 1
        elif use_cache:
            val_batches_data = shuffled[n_train_batches:]
            with torch.no_grad():
                for batch in val_batches_data:
                    agents_history = batch['agents_history'].to(device)
                    agents_mask = batch['agents_history_mask'].to(device)
                    agents_seq = batch['agents_seq'].to(device)
                    map_lanes = batch['map_lanes'].to(device) if 'map_lanes' in batch else None
                    map_lanes_mask = batch['map_lanes_mask'].to(device) if 'map_lanes_mask' in batch else None
                    map_polygons = batch['map_polygons'].to(device) if 'map_polygons' in batch else None
                    map_polygons_mask = batch['map_polygons_mask'].to(device) if 'map_polygons_mask' in batch else None

                    agents_pred = model(agents_history, agents_mask, map_lanes, map_lanes_mask,
                                        map_polygons=map_polygons, map_polygons_mask=map_polygons_mask)
                    loss = compute_transition_loss(agents_pred, agents_seq, agents_mask)
                    val_loss_sum += loss.item()
                    val_batches += 1
        else:
            # Fallback: no val split for slow DataLoader, use train loss
            val_loss_sum = epoch_loss
            val_batches = n_batches

        avg_val_loss = val_loss_sum / max(val_batches, 1)

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"train={avg_train_loss:.4f}  val={avg_val_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"({elapsed:.1f}s, {elapsed/max(n_batches,1):.2f}s/batch)")

        # LR scheduler based on VALIDATION loss
        scheduler.step(avg_val_loss)

        # Save checkpoint based on VALIDATION loss
        is_best = avg_val_loss < best_val_loss
        best_val_loss = min(avg_val_loss, best_val_loss)

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_val_loss,
        }
        ckpt_path = os.path.join(
            cfg.CHECKPOINT_DIR, f"stage_a_epoch_{epoch+1:03d}.pt"
        )
        torch.save(ckpt, ckpt_path)
        if is_best:
            best_path = os.path.join(cfg.CHECKPOINT_DIR, "stage_a_best.pt")
            torch.save(ckpt, best_path)
            print(f"  * New best (val) saved: {best_path}")

    print(f"\n[Stage A] Done. Best val L_tm={best_val_loss:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CarPlanner Stage A: Transition Model Pre-training"
    )
    p.add_argument('--split', default='mini',
                   choices=['mini', 'train_boston', 'train_pittsburgh', 'train_singapore',
                            'train_all', 'train_all_balanced'])
    p.add_argument('--epochs', type=int, default=cfg.EPOCHS)
    p.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE)
    p.add_argument('--num_workers', type=int, default=cfg.NUM_WORKERS,
                   help='Workers for slow DataLoader fallback only')
    p.add_argument('--max_per_file', type=int, default=None,
                   help='Cap on samples per DB file (slow fallback only)')
    p.add_argument('--resume', default=None,
                   help='Path to Stage A checkpoint to resume from')
    p.add_argument('--pin_gpu', action='store_true',
                   help='Load entire cache to GPU for zero-transfer training')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
