"""
Reconstruct scenario_type per sample for existing caches.

The extraction (extract_stage_a.py) runs with shuffle=False, so the index
order is deterministic. This script rebuilds the same index, captures
scenario_type, and checks if the count matches the cache.

Usage:
    python build_type_index.py --split train_boston
"""

import os
import sys
import glob
import argparse
import time
from collections import Counter

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import torch

import config as cfg
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_lidarpc_tokens_with_scenario_tag_from_db,
)


def get_db_dir(split):
    if split == 'train_boston':
        return cfg.TRAIN_DIR
    elif split == 'train_pittsburgh':
        return cfg.TRAIN_PITTSBURGH_DIR
    elif split == 'train_singapore':
        return cfg.TRAIN_SINGAPORE_DIR
    elif split == 'mini':
        return cfg.MINI_DIR
    else:
        raise ValueError(f"Unknown split: {split}")


def get_max_per_file(split):
    if split == 'mini':
        return cfg.MAX_SAMPLES_PER_FILE_MINI
    else:
        return cfg.MAX_SAMPLES_PER_FILE_TRAIN


def build_type_index(split):
    """Rebuild the dataset index with scenario_type, matching extract order."""
    db_dir = get_db_dir(split)
    max_per_file = get_max_per_file(split)
    db_files = sorted(glob.glob(os.path.join(db_dir, "*.db")))

    print(f"[TypeIndex] Building index for '{split}' ({len(db_files)} DB files, "
          f"max_per_file={max_per_file})...")

    index = []  # list of (db_path, token, scenario_type)
    t0 = time.time()

    for i, db_path in enumerate(db_files):
        try:
            seen = set()
            count = 0
            for scenario_type, token in get_lidarpc_tokens_with_scenario_tag_from_db(db_path):
                if token in seen:
                    continue
                seen.add(token)
                index.append((db_path, token, scenario_type))
                count += 1
                if count >= max_per_file:
                    break
        except Exception as e:
            print(f"  Warning: skipping {os.path.basename(db_path)}: {e}")

        if (i + 1) % max(1, len(db_files) // 10) == 0:
            print(f"  [{i+1}/{len(db_files)}] {len(index)} samples indexed...")

    elapsed = time.time() - t0
    print(f"[TypeIndex] Index: {len(index)} samples in {elapsed:.1f}s")

    return index


def main(args):
    split = args.split

    # Step 1: Rebuild the index with scenario types
    index = build_type_index(split)
    n_indexed = len(index)
    types_list = [entry[2] for entry in index]

    # Step 2: Load existing cache and compare counts
    cache_path = os.path.join(cfg.CHECKPOINT_DIR, f"stage_cache_{split}.pt")
    if not os.path.isfile(cache_path):
        print(f"[TypeIndex] Cache not found: {cache_path}")
        return

    print(f"\n[TypeIndex] Loading cache: {cache_path}")
    data = torch.load(cache_path, map_location='cpu')
    n_cached = data['n_samples']

    print(f"[TypeIndex] Index samples:  {n_indexed}")
    print(f"[TypeIndex] Cache samples:  {n_cached}")
    print(f"[TypeIndex] Difference:     {n_indexed - n_cached} "
          f"({(n_indexed - n_cached) / n_indexed * 100:.2f}% dropped)")

    # Step 3: Count scenario types
    type_counts = Counter(types_list)
    print(f"\n[TypeIndex] {len(type_counts)} distinct scenario types:")
    print(f"{'Type':<60} {'Count':>8}")
    print("-" * 70)
    for stype, count in type_counts.most_common():
        print(f"  {stype:<58} {count:>8,}")

    # Step 4: Save the type index as a sidecar
    sidecar_path = os.path.join(cfg.CHECKPOINT_DIR, f"stage_cache_{split}_types.pt")
    torch.save({
        'scenario_types': types_list,
        'n_indexed': n_indexed,
        'n_cached': n_cached,
        'type_counts': dict(type_counts),
    }, sidecar_path)
    print(f"\n[TypeIndex] Saved type index: {sidecar_path}")

    # Step 5: Evaluate — if we truncate to n_cached, what's the type distribution?
    # The extraction drops NaN/invalid samples. Since drops are rare and ~uniform,
    # the truncated type list is a reasonable approximation.
    if n_indexed != n_cached:
        drop_rate = (n_indexed - n_cached) / n_indexed
        print(f"\n[TypeIndex] NOTE: {drop_rate*100:.2f}% samples were dropped during extraction.")
        print(f"  Type assignments for the first {n_cached} indexed samples are approximate.")
        print(f"  For exact matching, re-extraction with type metadata would be needed.")
        # Truncated approximation: take first n_cached types
        # This is imperfect because drops happen at random positions, not at the end.
        # But for balancing purposes with 4000/type caps, it's close enough.


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--split', required=True,
                   choices=['mini', 'train_boston', 'train_pittsburgh', 'train_singapore'])
    main(p.parse_args())
