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
import random
from datetime import datetime

# Flush stdout immediately so nohup log updates in real time
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as cfg
from data_loader import make_dataloader, make_cached_dataloader
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

    # Data — cache-first, slow fallback
    cache_path = os.path.join(
        cfg.CHECKPOINT_DIR, f"stage_cache_{args.split}.pt"
    )
    val_cache_path = os.path.join(cfg.CHECKPOINT_DIR, "stage_cache_val14.pt")
    use_cache = os.path.isfile(cache_path)
    pin_gpu = args.pin_gpu and device.type == 'cuda'

    if use_cache and pin_gpu:
        print(f"[Train] Loading cache to GPU (pin_gpu): {cache_path}")
        t0 = time.time()
        data = torch.load(cache_path, map_location=device)
        gpu_cache = {
            'agents_now':          data['agents_now'],
            'agents_history':      data['agents_history'],
            'agents_mask':         data['agents_mask'],
            'agents_seq':          data['agents_seq'],
            'gt_trajectory':       data['gt_trajectory'],
            'mode_label':          data['mode_label'],
            'map_lanes':           data['map_lanes'],
            'map_lanes_mask':      data['map_lanes_mask'],
            'map_polygons':        data.get('map_polygons', torch.zeros(data['n_samples'], cfg.N_POLYGONS, cfg.N_LANE_POINTS, cfg.D_POLYGON_POINT, device=device)),
            'map_polygons_mask':   data.get('map_polygons_mask', torch.zeros(data['n_samples'], cfg.N_POLYGONS, device=device)),
            'route_polylines':     data.get('route_polylines', torch.zeros(data['n_samples'], cfg.N_LAT, cfg.N_ROUTE_POINTS, cfg.D_POLYLINE_POINT, device=device)),
            'route_mask':          data.get('route_mask', torch.zeros(data['n_samples'], cfg.N_LAT, device=device)),
        }
        n_samples = data['n_samples']
        loader = None
        all_batches = None
        print(f"[Train] GPU cache: {n_samples} samples in "
              f"{time.time()-t0:.1f}s, ~{torch.cuda.memory_allocated()/1e9:.1f}GB GPU used")
    elif use_cache:
        print(f"[Train] Loading pre-extracted cache: {cache_path}")
        loader = make_cached_dataloader(
            cache_path,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        all_batches = list(loader)
        n_samples = len(loader.dataset)
        print(f"[Train] Cache loaded: {n_samples} samples, "
              f"{len(all_batches)} batches")
    else:
        print(f"[Train] WARNING: No cache found. Falling back to slow DataLoader.")
        loader = make_dataloader(
            args.split,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            max_per_file=args.max_per_file,
        )
        all_batches = None
        n_samples = len(loader.dataset)
        print(f"[Train] Dataset size: {n_samples}, "
              f"Batches/epoch: {len(loader)}")

    # Model
    cfg.set_stage('b')
    model = CarPlanner().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Parameters: {n_params:,}")

    # Load frozen transition model from Stage A if provided
    if args.transition_ckpt:
        model.load_transition_model(args.transition_ckpt, freeze=True)
        print(f"[Train] Loaded frozen transition model from: {args.transition_ckpt}")

    # Validation: load val14 cache to GPU unconditionally (small, ~73 MB for 1,116 samples)
    val_gpu_cache = None
    n_val_samples = 0
    n_val = 0
    val_idx = None
    if os.path.isfile(val_cache_path):
        print(f"[Train] Loading val14 cache to GPU: {val_cache_path}")
        val_data = torch.load(val_cache_path, map_location=device)
        val_gpu_cache = {
            'agents_now':          val_data['agents_now'],
            'agents_history':      val_data['agents_history'],
            'agents_mask':         val_data['agents_mask'],
            'agents_seq':          val_data['agents_seq'],
            'gt_trajectory':       val_data['gt_trajectory'],
            'mode_label':          val_data['mode_label'],
            'map_lanes':           val_data['map_lanes'],
            'map_lanes_mask':      val_data['map_lanes_mask'],
            'map_polygons':        val_data.get('map_polygons', torch.zeros(val_data['n_samples'], cfg.N_POLYGONS, cfg.N_LANE_POINTS, cfg.D_POLYGON_POINT, device=device)),
            'map_polygons_mask':   val_data.get('map_polygons_mask', torch.zeros(val_data['n_samples'], cfg.N_POLYGONS, device=device)),
            'route_polylines':     val_data.get('route_polylines', torch.zeros(val_data['n_samples'], cfg.N_LAT, cfg.N_ROUTE_POINTS, cfg.D_POLYLINE_POINT, device=device)),
            'route_mask':          val_data.get('route_mask', torch.zeros(val_data['n_samples'], cfg.N_LAT, device=device)),
        }
        n_val_samples = val_data['n_samples']
        del val_data
        print(f"[Train] Val (val14): {n_val_samples} samples on GPU")

    if pin_gpu:
        n_train = n_samples
        train_idx = torch.arange(n_train, device=device)
        if val_gpu_cache is None:
            print(f"[Train] WARNING: val14 cache not found at {val_cache_path}, falling back to 90/10 split")
            n_val = max(1, int(n_samples * 0.1))
            n_train = n_samples - n_val
            split_perm = torch.randperm(n_samples, device=device)
            train_idx = split_perm[:n_train]
            val_idx = split_perm[n_train:]
            print(f"[Train] Split: {n_train} train, {n_val} val")

    # Pre-compute β outputs for all samples once, store on CPU RAM (1.67 GB).
    # β is frozen + inputs are fixed → output is identical every epoch.
    # Per-batch: slice CPU tensor → transfer ~200KB to GPU → no redundant recomputes.
    beta_seq_cpu = None
    if args.transition_ckpt and pin_gpu:
        print(f"[Train] Pre-computing β outputs for {n_samples} samples (stored on CPU)...")
        t0 = time.time()
        model.eval()
        precomp_bs = 512
        chunks = []
        with torch.no_grad():
            for start in range(0, n_samples, precomp_bs):
                end = min(start + precomp_bs, n_samples)
                out = model.transition_model(
                    gpu_cache['agents_history'][start:end],
                    gpu_cache['agents_mask'][start:end],
                    gpu_cache['map_lanes'][start:end],
                    gpu_cache['map_lanes_mask'][start:end],
                    map_polygons=gpu_cache['map_polygons'][start:end],
                    map_polygons_mask=gpu_cache['map_polygons_mask'][start:end],
                )                                          # (chunk, T, N, Da) on GPU
                chunks.append(out.cpu())                   # move to CPU immediately
                del out
        beta_seq_cpu = torch.cat(chunks, dim=0)            # (N, T, N_agents, Da) on CPU
        model._transition_loaded = False                   # disable per-batch β — we cached it
        model.train()
        size_gb = beta_seq_cpu.element_size() * beta_seq_cpu.nelement() / 1e9
        print(f"[Train] β pre-computation done in {time.time()-t0:.1f}s  "
              f"({size_gb:.2f} GB on CPU RAM)")

    # Pre-compute β for val14 cache too (only when train β was pre-computed, i.e. pin_gpu path)
    val_beta_seq_cpu = None
    if beta_seq_cpu is not None and val_gpu_cache is not None:
        print(f"[Train] Pre-computing β outputs for {n_val_samples} val samples...")
        model.eval()
        model._transition_loaded = True
        chunks = []
        precomp_bs = 512
        with torch.no_grad():
            for start in range(0, n_val_samples, precomp_bs):
                end = min(start + precomp_bs, n_val_samples)
                out = model.transition_model(
                    val_gpu_cache['agents_history'][start:end],
                    val_gpu_cache['agents_mask'][start:end],
                    val_gpu_cache['map_lanes'][start:end],
                    val_gpu_cache['map_lanes_mask'][start:end],
                    map_polygons=val_gpu_cache['map_polygons'][start:end],
                    map_polygons_mask=val_gpu_cache['map_polygons_mask'][start:end],
                )
                chunks.append(out.cpu())
                del out
        val_beta_seq_cpu = torch.cat(chunks, dim=0)
        model._transition_loaded = False
        model.train()
        print(f"[Train] Val β pre-computation done ({val_beta_seq_cpu.shape})")
    elif beta_seq_cpu is None:
        val_beta_seq_cpu = None

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

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg.CHECKPOINT_DIR, f"stage_b_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Train] Checkpoints → {run_dir}")

    best_val_loss = float('inf')
    best_train_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {'L_CE': 0., 'L_side': 0., 'L_gen': 0., 'L_total': 0.}
        t0 = time.time()

        if pin_gpu:
            train_perm = torch.randperm(n_train, device=device)
            n_batches_epoch = n_train // args.batch_size

        def get_batch(batch_idx, batch=None, indices=None):
            if pin_gpu:
                if indices is not None:
                    idx = indices
                else:
                    idx = train_idx[train_perm[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]]
                if beta_seq_cpu is not None:
                    agents_seq_batch = beta_seq_cpu[idx.cpu()].to(device)
                else:
                    agents_seq_batch = gpu_cache['agents_seq'][idx]
                return (gpu_cache['agents_now'][idx],
                        gpu_cache['agents_history'][idx],
                        gpu_cache['agents_mask'][idx],
                        agents_seq_batch,
                        gpu_cache['gt_trajectory'][idx],
                        gpu_cache['mode_label'][idx],
                        gpu_cache['map_lanes'][idx],
                        gpu_cache['map_lanes_mask'][idx],
                        gpu_cache['map_polygons'][idx],
                        gpu_cache['map_polygons_mask'][idx],
                        gpu_cache['route_polylines'][idx],
                        gpu_cache['route_mask'][idx])
            else:
                return (batch['agents_now'].to(device),
                        batch['agents_history'].to(device),
                        batch['agents_history_mask'].to(device),
                        batch['agents_seq'].to(device),
                        batch['gt_trajectory'].to(device),
                        batch['mode_label'].to(device),
                        batch['map_lanes'].to(device),
                        batch['map_lanes_mask'].to(device),
                        batch['map_polygons'].to(device) if 'map_polygons' in batch else None,
                        batch['map_polygons_mask'].to(device) if 'map_polygons_mask' in batch else None,
                        batch['route_polylines'].to(device) if 'route_polylines' in batch else None,
                        batch['route_mask'].to(device) if 'route_mask' in batch else None)

        cpu_batches = random.sample(all_batches, len(all_batches)) if (use_cache and not pin_gpu) else (loader if not pin_gpu else [])
        outer = range(n_batches_epoch) if pin_gpu else enumerate(cpu_batches)

        for item in outer:
            if pin_gpu:
                batch_idx = item
                agents_now, agents_history, agents_mask, agents_seq, \
                    gt_traj, mode_label, map_lanes, map_lanes_mask, \
                    map_polygons, map_polygons_mask, \
                    route_polylines, route_mask_batch = get_batch(batch_idx)
            else:
                batch_idx, batch = item
                agents_now, agents_history, agents_mask, agents_seq, \
                    gt_traj, mode_label, map_lanes, map_lanes_mask, \
                    map_polygons, map_polygons_mask, \
                    route_polylines, route_mask_batch = get_batch(batch_idx, batch)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                mode_logits, side_traj, pred_traj = model.forward_train(
                    agents_now, agents_mask, agents_seq, gt_traj, mode_label,
                    map_lanes=map_lanes, map_lanes_mask=map_lanes_mask,
                    agents_history=agents_history,
                    map_polygons=map_polygons, map_polygons_mask=map_polygons_mask,
                    route_polylines=route_polylines, route_mask=route_mask_batch,
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

            n_batches_total = n_batches_epoch if pin_gpu else len(cpu_batches)
            if (batch_idx + 1) % max(1, n_batches_total // 5) == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}  "
                      f"[{batch_idx+1}/{n_batches_total}]  "
                      f"L_total={loss_dict['L_total']:.4f}  "
                      f"L_CE={loss_dict['L_CE']:.4f}  "
                      f"L_gen={loss_dict['L_gen']:.4f}")

        # Epoch train summary
        n = n_batches_epoch if pin_gpu else len(cpu_batches)
        avg_train = {k: v / n for k, v in epoch_losses.items()}
        elapsed = time.time() - t0

        # Validation
        model.eval()
        val_losses = {'L_CE': 0., 'L_side': 0., 'L_gen': 0., 'L_total': 0.}
        val_batches = 0

        if val_gpu_cache is not None:
            n_val_batches = (n_val_samples + args.batch_size - 1) // args.batch_size
            with torch.no_grad():
                for b in range(n_val_batches):
                    s = b * args.batch_size
                    e = min(s + args.batch_size, n_val_samples)
                    agents_now = val_gpu_cache['agents_now'][s:e]
                    agents_history = val_gpu_cache['agents_history'][s:e]
                    agents_mask = val_gpu_cache['agents_mask'][s:e]
                    agents_seq = val_gpu_cache['agents_seq'][s:e]
                    gt_traj = val_gpu_cache['gt_trajectory'][s:e]
                    mode_label = val_gpu_cache['mode_label'][s:e]
                    map_lanes = val_gpu_cache['map_lanes'][s:e]
                    map_lanes_mask = val_gpu_cache['map_lanes_mask'][s:e]
                    map_polygons = val_gpu_cache['map_polygons'][s:e]
                    map_polygons_mask = val_gpu_cache['map_polygons_mask'][s:e]
                    route_polylines = val_gpu_cache['route_polylines'][s:e]
                    route_mask_batch = val_gpu_cache['route_mask'][s:e]

                    if beta_seq_cpu is not None:
                        agents_seq = val_beta_seq_cpu[s:e].to(device) if val_beta_seq_cpu is not None else agents_seq

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        mode_logits, side_traj, pred_traj = model.forward_train(
                            agents_now, agents_mask, agents_seq, gt_traj, mode_label,
                            map_lanes=map_lanes, map_lanes_mask=map_lanes_mask,
                            agents_history=agents_history,
                            map_polygons=map_polygons, map_polygons_mask=map_polygons_mask,
                            route_polylines=route_polylines, route_mask=route_mask_batch,
                        )
                        _, loss_dict = compute_il_loss(
                            mode_logits, side_traj, pred_traj, gt_traj, mode_label
                        )
                    for k in val_losses:
                        val_losses[k] += loss_dict[k]
                    val_batches += 1
        elif pin_gpu:
            # Fallback 90/10 path (no val14 cache)
            n_val_batches = max(1, n_val // args.batch_size)
            with torch.no_grad():
                for b in range(n_val_batches):
                    idx = val_idx[b * args.batch_size : (b + 1) * args.batch_size]
                    agents_now, agents_history, agents_mask, agents_seq, \
                        gt_traj, mode_label, map_lanes, map_lanes_mask, \
                        map_polygons, map_polygons_mask, \
                        route_polylines, route_mask_batch = get_batch(b, indices=idx)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        mode_logits, side_traj, pred_traj = model.forward_train(
                            agents_now, agents_mask, agents_seq, gt_traj, mode_label,
                            map_lanes=map_lanes, map_lanes_mask=map_lanes_mask,
                            agents_history=agents_history,
                            map_polygons=map_polygons, map_polygons_mask=map_polygons_mask,
                        )
                        _, loss_dict = compute_il_loss(
                            mode_logits, side_traj, pred_traj, gt_traj, mode_label
                        )
                    for k in val_losses:
                        val_losses[k] += loss_dict[k]
                    val_batches += 1
        else:
            val_losses = avg_train.copy()
            val_batches = 1

        avg_val = {k: v / max(val_batches, 1) for k, v in val_losses.items()}

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"train={avg_train['L_total']:.4f}  val={avg_val['L_total']:.4f}  "
              f"L_CE={avg_val['L_CE']:.4f}  "
              f"L_side={avg_val['L_side']:.4f}  "
              f"L_gen={avg_val['L_gen']:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"({elapsed:.1f}s)")

        scheduler.step(avg_val['L_total'])

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': avg_train,
            'val_loss': avg_val,
        }
        ckpt_path = os.path.join(run_dir, f"stage_b_epoch_{epoch+1:03d}.pt")
        torch.save(ckpt, ckpt_path)

        # Best val checkpoint
        is_best_val = avg_val['L_total'] < best_val_loss
        best_val_loss = min(avg_val['L_total'], best_val_loss)
        if is_best_val:
            best_path = os.path.join(run_dir, "stage_b_best.pt")
            torch.save(ckpt, best_path)
            torch.save(ckpt, os.path.join(cfg.CHECKPOINT_DIR, "stage_b_best.pt"))
            print(f"  * New best (val)   L_total={avg_val['L_total']:.4f} → {best_path}")

        # Best train checkpoint
        is_best_train = avg_train['L_total'] < best_train_loss
        best_train_loss = min(avg_train['L_total'], best_train_loss)
        if is_best_train:
            best_train_path = os.path.join(run_dir, "stage_b_best_train.pt")
            torch.save(ckpt, best_train_path)
            torch.save(ckpt, os.path.join(cfg.CHECKPOINT_DIR, "stage_b_best_train.pt"))
            print(f"  * New best (train) L_total={avg_train['L_total']:.4f} → {best_train_path}")

    print(f"\n[Train] Done. Best val L_total={best_val_loss:.4f}  Best train L_total={best_train_loss:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CarPlanner IL training")
    p.add_argument('--split', default='mini',
                   choices=['mini', 'train_boston', 'train_pittsburgh', 'train_singapore',
                            'train_all', 'train_all_balanced', 'train_3city_balanced',
                            'train_4city_balanced',
                            'train_boston_balanced', 'train_pittsburgh_balanced',
                            'train_singapore_balanced'])
    p.add_argument('--epochs', type=int, default=cfg.EPOCHS)
    p.add_argument('--batch_size', type=int, default=96)
    p.add_argument('--num_workers', type=int, default=cfg.NUM_WORKERS)
    p.add_argument('--max_per_file', type=int, default=None,
                   help='Cap on samples per DB file (default: from config)')
    p.add_argument('--resume', default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--transition_ckpt', default=None,
                   help='Path to Stage A checkpoint (loads frozen transition model)')
    p.add_argument('--pin_gpu', action='store_true',
                   help='Load entire cache to GPU for faster training')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
