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
import numpy as np

import config as cfg
from data_loader import NuPlanCarPlannerDataset


def extract(args):
    print(f"[Extract] Building dataset index for '{args.split}'...")
    dataset = NuPlanCarPlannerDataset(args.split, max_per_file=args.max_per_file)
    n_samples = len(dataset)
    print(f"[Extract] {n_samples} samples to extract")

    if n_samples == 0:
        print("[Extract] No samples found. Check your DB files and config paths.")
        return

    # Pre-allocate arrays
    agents_history_buf = np.zeros(
        (n_samples, cfg.T_HIST, cfg.N_AGENTS, cfg.D_AGENT), dtype=np.float32
    )
    agents_mask_buf = np.zeros((n_samples, cfg.N_AGENTS), dtype=np.float32)
    agents_seq_buf = np.zeros(
        (n_samples, cfg.T_FUTURE, cfg.N_AGENTS, cfg.D_AGENT), dtype=np.float32
    )
    agents_now_buf = np.zeros(
        (n_samples, cfg.N_AGENTS, cfg.D_AGENT), dtype=np.float32
    )
    gt_trajectory_buf = np.zeros(
        (n_samples, cfg.T_FUTURE, 3), dtype=np.float32
    )
    mode_label_buf = np.zeros(n_samples, dtype=np.int64)
    map_lanes_buf = np.zeros(
        (n_samples, cfg.N_LANES, cfg.N_LANE_POINTS, cfg.D_POLYLINE_POINT), dtype=np.float32
    )
    map_lanes_mask_buf = np.zeros((n_samples, cfg.N_LANES), dtype=np.float32)

    valid_count = 0
    t0 = time.time()

    for i in range(n_samples):
        sample = dataset[i]

        # Skip completely empty samples (no valid agents)
        if sample['agents_history_mask'].sum() < 0.5:
            continue

        agents_history_buf[valid_count] = sample['agents_history'].numpy()
        agents_mask_buf[valid_count] = sample['agents_history_mask'].numpy()
        agents_seq_buf[valid_count] = sample['agents_seq'].numpy()
        agents_now_buf[valid_count] = sample['agents_now'].numpy()
        gt_trajectory_buf[valid_count] = sample['gt_trajectory'].numpy()
        mode_label_buf[valid_count] = sample['mode_label'].numpy().item() if sample['mode_label'].dim() == 0 else sample['mode_label'].numpy()
        map_lanes_buf[valid_count] = sample['map_lanes'].numpy()
        map_lanes_mask_buf[valid_count] = sample['map_lanes_mask'].numpy()
        valid_count += 1

        if (valid_count) % max(1, n_samples // 10) == 0:
            elapsed = time.time() - t0
            rate = valid_count / elapsed
            eta = (n_samples - i) / rate if rate > 0 else 0
            print(f"  [{valid_count}/{n_samples}]  "
                  f"{rate:.1f} samples/s  "
                  f"ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"[Extract] Extracted {valid_count}/{n_samples} valid samples in {elapsed:.1f}s")

    if valid_count == 0:
        print("[Extract] No valid samples. Check your data.")
        return

    # Trim to valid count
    agents_history_buf = agents_history_buf[:valid_count]
    agents_mask_buf = agents_mask_buf[:valid_count]
    agents_seq_buf = agents_seq_buf[:valid_count]
    agents_now_buf = agents_now_buf[:valid_count]
    gt_trajectory_buf = gt_trajectory_buf[:valid_count]
    mode_label_buf = mode_label_buf[:valid_count]
    map_lanes_buf = map_lanes_buf[:valid_count]
    map_lanes_mask_buf = map_lanes_mask_buf[:valid_count]

    # Save
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    cache_path = os.path.join(cfg.CHECKPOINT_DIR, f"stage_cache_{args.split}.pt")

    data = {
        'agents_history': torch.from_numpy(agents_history_buf),
        'agents_mask': torch.from_numpy(agents_mask_buf),
        'agents_seq': torch.from_numpy(agents_seq_buf),
        'agents_now': torch.from_numpy(agents_now_buf),
        'gt_trajectory': torch.from_numpy(gt_trajectory_buf),
        'mode_label': torch.from_numpy(mode_label_buf),
        'map_lanes': torch.from_numpy(map_lanes_buf),
        'map_lanes_mask': torch.from_numpy(map_lanes_mask_buf),
        'n_samples': valid_count,
        'split': args.split,
    }

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
    print(f"  map_lanes:      {tuple(data['map_lanes'].shape)}")
    print(f"  map_lanes_mask: {tuple(data['map_lanes_mask'].shape)}")
    print(f"  Avg valid agents: {data['agents_mask'].sum(1).mean():.1f}")
    print(f"  Avg valid lanes:  {data['map_lanes_mask'].sum(1).mean():.1f}")
    print(f"  agents_history range: [{data['agents_history'].min():.2f}, {data['agents_history'].max():.2f}]")
    print(f"  gt_trajectory range:  [{data['gt_trajectory'].min():.2f}, {data['gt_trajectory'].max():.2f}]")
    print(f"  mode_label range:     [{data['mode_label'].min()}, {data['mode_label'].max()}]")

    has_nan = any(t.isnan().any() for t in [
        data['agents_history'], data['agents_seq'],
        data['gt_trajectory'], data['map_lanes']
    ])
    print(f"  NaN check: {'FAIL' if has_nan else 'OK'}")


def parse_args():
    p = argparse.ArgumentParser(description="Extract data from SQLite to .pt cache (all stages)")
    p.add_argument('--split', default='mini', choices=['mini', 'train_boston'])
    p.add_argument('--max_per_file', type=int, default=None)
    return p.parse_args()


if __name__ == '__main__':
    extract(parse_args())
