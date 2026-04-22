"""
Merge per-city .pt caches into a single master cache for training.

Usage:
    python merge_caches.py
    python merge_caches.py --cities train_boston train_pittsburgh train_singapore
"""

import os
import argparse
import torch

import config as cfg


def merge(args):
    cities = args.cities
    cache_dir = cfg.CHECKPOINT_DIR

    print(f"[Merge] Loading caches for: {cities}")
    caches = []
    total_samples = 0

    for city in cities:
        path = os.path.join(cache_dir, f"stage_cache_{city}.pt")
        if not os.path.isfile(path):
            print(f"  [SKIP] {city}: {path} not found")
            continue
        data = torch.load(path, map_location='cpu')
        n = data['n_samples']
        print(f"  {city}: {n} samples ({os.path.getsize(path)/1e6:.1f} MB)")
        caches.append(data)
        total_samples += n

    if not caches:
        print("[Merge] No caches found. Nothing to merge.")
        return

    print(f"\n[Merge] Merging {len(caches)} caches ({total_samples} total samples)...")

    keys = ['agents_history', 'agents_mask', 'agents_seq', 'agents_now',
            'gt_trajectory', 'mode_label', 'map_lanes', 'map_lanes_mask',
            'map_polygons', 'map_polygons_mask']

    merged = {}
    for k in keys:
        merged[k] = torch.cat([c[k] for c in caches], dim=0)
        print(f"  {k}: {tuple(merged[k].shape)}")

    merged['n_samples'] = total_samples
    merged['split'] = 'train_all'

    out_path = os.path.join(cache_dir, f"stage_cache_{args.output}.pt")
    print(f"\n[Merge] Saving to {out_path}...")
    torch.save(merged, out_path)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[Merge] Done. {total_samples} samples, {size_mb:.1f} MB")


def parse_args():
    p = argparse.ArgumentParser(description="Merge per-city caches into one")
    p.add_argument('--cities', nargs='+',
                   default=['train_boston', 'train_pittsburgh', 'train_singapore'],
                   help='City splits to merge')
    p.add_argument('--output', default='train_all',
                   help='Output cache name (default: train_all)')
    return p.parse_args()


if __name__ == '__main__':
    merge(parse_args())
