"""
CarPlanner — One-time data extraction from SQLite to fast .pt cache.

Extracts all 7 tensors needed by Stages A, B, and C:
  - agents_history:  (N, H, Da)       — real past frames
  - agents_mask:     (N,)             — agent validity mask
  - agents_seq:      (T, N, Da)       — GT agent futures
  - agents_now:      (N, Da)          — agents at t=0 (= history[:, -1])
  - gt_trajectory:   (T, 3)           — ego GT future waypoints
  - mode_label:      ()               — c* mode assignment
  - map_lanes:       (N_LANES, N_PTS, 27)  — lane polylines
  - map_lanes_mask:  (N_LANES,)       — lane validity mask

Usage:
    python extract_stage_a.py --split mini
    python extract_stage_a.py --split mini --max_per_file 500
"""

import os
import sys
import argparse
import time

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import torch
from torch.utils.data import DataLoader
import numpy as np

import config as cfg
from data_loader import NuPlanCarPlannerDataset


def _safe_collate(batch):
    """Drop None samples (failed DB reads) before stacking."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def extract(args):
    print(f"[Extract] Building dataset index for '{args.split}'...")
    dataset = NuPlanCarPlannerDataset(args.split, max_per_file=args.max_per_file)
    n_samples = len(dataset)
    print(f"[Extract] {n_samples} samples to extract")
    print(f"[Extract] Workers: {args.num_workers}  Batch size: {args.batch_size}")

    if n_samples == 0:
        print("[Extract] No samples found. Check your DB files and config paths.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_safe_collate,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    keys = ['ego_history',
            'agents_history', 'agents_history_mask', 'agents_seq', 'agents_now',
            'gt_trajectory', 'mode_label', 'map_lanes', 'map_lanes_mask',
            'map_polygons', 'map_polygons_mask',
            'route_polylines', 'route_mask']
    buffers = {k: [] for k in keys}

    valid_count = 0
    t0 = time.time()
    log_interval = max(1, len(loader) // 20)

    nan_dropped = 0
    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue
        B = batch['agents_history_mask'].shape[0]
        # Drop samples with no valid agents
        keep = batch['agents_history_mask'].sum(dim=1) > 0.5
        # Drop samples containing NaN in any numeric tensor
        nan_check_keys = ['agents_history', 'agents_seq', 'gt_trajectory', 'map_lanes', 'map_polygons']
        for k in nan_check_keys:
            has_nan = torch.isnan(batch[k]).view(B, -1).any(dim=1)
            keep = keep & ~has_nan
        nan_dropped += (B - keep.sum().item())
        if not keep.any():
            continue
        for k in keys:
            buffers[k].append(batch[k][keep])
        valid_count += keep.sum().item()

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - t0
            rate = valid_count / elapsed
            eta = (n_samples - valid_count) / rate if rate > 0 else 0
            print(f"  [{valid_count}/{n_samples}]  "
                  f"{rate:.1f} samples/s  "
                  f"ETA: {eta/60:.1f}min")

    elapsed = time.time() - t0
    print(f"[Extract] Extracted {valid_count}/{n_samples} valid samples in "
          f"{elapsed/60:.1f}min  ({valid_count/max(elapsed,1):.1f} samples/s)")
    if nan_dropped > 0:
        print(f"[Extract] Dropped {nan_dropped} samples containing NaN")

    if valid_count == 0:
        print("[Extract] No valid samples. Check your data.")
        return

    # Concatenate collected batches
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    cache_path = os.path.join(cfg.CHECKPOINT_DIR, f"stage_cache_{args.split}.pt")

    data = {k: torch.cat(buffers[k], dim=0) for k in keys}
    # Rename mask key to match training scripts
    data['agents_mask'] = data.pop('agents_history_mask')
    data['n_samples'] = valid_count
    data['split'] = args.split

    print(f"[Extract] Saving to {cache_path}...")
    torch.save(data, cache_path)

    size_mb = os.path.getsize(cache_path) / 1e6
    print(f"[Extract] Done. {valid_count} samples, {size_mb:.1f} MB")

    # Quick sanity check
    print(f"\n[Extract] Sanity check:")
    print(f"  agents_history: {tuple(data['agents_history'].shape)}")
    print(f"  agents_mask:    {tuple(data['agents_mask'].shape)}")
    print(f"  agents_seq:     {tuple(data['agents_seq'].shape)}")
    print(f"  agents_now:     {tuple(data['agents_now'].shape)}")
    print(f"  gt_trajectory:  {tuple(data['gt_trajectory'].shape)}")
    print(f"  mode_label:     {tuple(data['mode_label'].shape)}")
    print(f"  map_lanes:        {tuple(data['map_lanes'].shape)}")
    print(f"  map_lanes_mask:   {tuple(data['map_lanes_mask'].shape)}")
    print(f"  map_polygons:     {tuple(data['map_polygons'].shape)}")
    print(f"  map_polygons_mask:{tuple(data['map_polygons_mask'].shape)}")
    print(f"  route_polylines:  {tuple(data['route_polylines'].shape)}")
    print(f"  route_mask:       {tuple(data['route_mask'].shape)}")
    print(f"  Avg valid agents:   {data['agents_mask'].sum(1).mean():.1f}")
    print(f"  Avg valid lanes:    {data['map_lanes_mask'].sum(1).mean():.1f}")
    print(f"  Avg valid polygons: {data['map_polygons_mask'].sum(1).mean():.1f}")
    print(f"  Avg valid routes:   {data['route_mask'].sum(1).mean():.1f}")
    print(f"  agents_history range: [{data['agents_history'].min():.2f}, {data['agents_history'].max():.2f}]")
    print(f"  gt_trajectory range:  [{data['gt_trajectory'].min():.2f}, {data['gt_trajectory'].max():.2f}]")
    print(f"  mode_label range:     [{data['mode_label'].min()}, {data['mode_label'].max()}]")

    has_nan = any(t.isnan().any() for t in [
        data['agents_history'], data['agents_seq'],
        data['gt_trajectory'], data['map_lanes'], data['map_polygons']
    ])
    print(f"  NaN check: {'FAIL' if has_nan else 'OK'}")


def parse_args():
    p = argparse.ArgumentParser(description="Extract data from SQLite to .pt cache (all stages)")
    p.add_argument('--split', default='mini',
                   choices=['mini', 'train_boston', 'train_pittsburgh', 'train_singapore',
                            'val14', 'test14_random', 'reduced_val14'])
    p.add_argument('--max_per_file', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=24,
                   help='Parallel CPU workers for DB reads (default: 24)')
    p.add_argument('--batch_size', type=int, default=64,
                   help='Batch size per worker (default: 64)')
    return p.parse_args()


if __name__ == '__main__':
    extract(parse_args())
