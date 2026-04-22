"""
CarPlanner — Balanced extraction matching PDM's train150k_split protocol.

Phase 1: Scan all DB files across cities, build full scenario_type→(db,token) index.
Phase 2: Sample up to N per type (default 4000), extract per-city sequentially
         (avoids cross-disk I/O thrashing), save one cache per city.

After extraction, merge with:  python merge_balanced.py

Usage:
    python extract_balanced.py --cap 4000
    python extract_balanced.py --cap 4000 --dry_run
    python extract_balanced.py --cap 4000 --num_workers 12
"""

import os
import sys
import argparse
import glob
import json
import time
import random
from collections import defaultdict, Counter

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import config as cfg
from data_loader import _load_sample
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_lidarpc_tokens_with_scenario_tag_from_db,
)


CITY_DIRS = {
    'train_boston': cfg.TRAIN_DIR,
    'train_pittsburgh': cfg.TRAIN_PITTSBURGH_DIR,
    'train_singapore': cfg.TRAIN_SINGAPORE_DIR,
    'train_vegas': cfg.TRAIN_VEGAS_DIR,
}


# ── Phase 1: Full type index ─────────────────────────────────────────────────

def build_full_index(cities):
    """Scan all DBs, return type→[(city, db_path, token)] and per-city index."""
    type_to_samples = defaultdict(list)
    total = 0

    for city, db_dir in cities.items():
        db_files = sorted(glob.glob(os.path.join(db_dir, "*.db")))
        city_count = 0
        t0 = time.time()
        print(f"[Phase 1] Scanning {city}: {len(db_files)} DB files...")

        for i, db_path in enumerate(db_files):
            try:
                seen = set()
                for scenario_type, token in get_lidarpc_tokens_with_scenario_tag_from_db(db_path):
                    if token in seen:
                        continue
                    seen.add(token)
                    type_to_samples[scenario_type].append((city, db_path, token))
                    city_count += 1
            except Exception as e:
                pass

            if (i + 1) % max(1, len(db_files) // 5) == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(db_files)}] {city_count:,} tokens, "
                      f"{elapsed:.1f}s")

        total += city_count
        elapsed = time.time() - t0
        print(f"  {city}: {city_count:,} tokens in {elapsed:.1f}s")

    print(f"\n[Phase 1] Total: {total:,} tokens across {len(type_to_samples)} types")
    return type_to_samples


# ── Phase 2: Per-city extraction ─────────────────────────────────────────────

class SelectedSamplesDataset(Dataset):
    """Dataset that extracts only pre-selected (db_path, token) pairs."""

    def __init__(self, selected_pairs, scenario_types):
        self._pairs = selected_pairs
        self._types = scenario_types

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        db_path, token = self._pairs[idx]
        sample = _load_sample(db_path, token)

        if sample is None:
            return None

        result = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v)
                  for k, v in sample.items()}
        result['_type_idx'] = torch.tensor(idx, dtype=torch.long)
        return result


def _safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def extract_city(city, pairs, types_list, args):
    """Extract features for one city's selected samples."""
    n_total = len(pairs)
    print(f"\n[{city}] Extracting {n_total:,} samples "
          f"(workers={args.num_workers}, batch={args.batch_size})...")

    dataset = SelectedSamplesDataset(pairs, types_list)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_safe_collate,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    keys = ['agents_history', 'agents_history_mask', 'agents_seq', 'agents_now',
            'gt_trajectory', 'mode_label', 'map_lanes', 'map_lanes_mask',
            'map_polygons', 'map_polygons_mask']
    # Preallocate output tensors (lazy-init on first valid batch) to avoid the
    # 2x memory spike that torch.cat over per-batch buffer lists caused before.
    out = None
    type_buffer = []

    valid_count = 0
    nan_dropped = 0
    t0 = time.time()
    log_interval = max(1, len(loader) // 10)

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue

        B = batch['agents_history_mask'].shape[0]
        type_indices = batch['_type_idx']

        keep = batch['agents_history_mask'].sum(dim=1) > 0.5
        nan_check_keys = ['agents_history', 'agents_seq', 'gt_trajectory',
                          'map_lanes', 'map_polygons']
        for k in nan_check_keys:
            has_nan = torch.isnan(batch[k]).view(B, -1).any(dim=1)
            keep = keep & ~has_nan
        nan_dropped += (B - keep.sum().item())

        if not keep.any():
            continue

        if out is None:
            out = {k: torch.empty((n_total,) + tuple(batch[k].shape[1:]),
                                  dtype=batch[k].dtype) for k in keys}

        n_keep = int(keep.sum().item())
        end = valid_count + n_keep
        for k in keys:
            out[k][valid_count:end] = batch[k][keep]

        kept_indices = type_indices[keep].tolist()
        for idx in kept_indices:
            type_buffer.append(types_list[idx])
        valid_count = end

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - t0
            rate = valid_count / max(elapsed, 1)
            eta = (n_total - valid_count) / max(rate, 1)
            print(f"  [{city}] [{valid_count:,}/{n_total:,}]  "
                  f"{rate:.1f} samples/s  ETA: {eta/60:.1f}min")

    elapsed = time.time() - t0
    print(f"[{city}] Extracted {valid_count:,}/{n_total:,} "
          f"in {elapsed/60:.1f}min ({valid_count/max(elapsed,1):.1f} samples/s)")
    if nan_dropped > 0:
        print(f"[{city}] Dropped {nan_dropped:,} samples (NaN/invalid)")

    if valid_count == 0:
        return None, None

    # Trim preallocated tail. Clone + drop each key one-by-one so peak memory
    # stays ~constant (old oversized tensor is freed as its trimmed copy lands).
    data = {}
    for k in keys:
        data[k] = out[k][:valid_count].clone()
        out[k] = None
    data['agents_mask'] = data.pop('agents_history_mask')
    data['n_samples'] = valid_count
    data['split'] = city

    return data, type_buffer


def extract(args):
    random.seed(args.seed)
    cities = CITY_DIRS
    if args.cities:
        cities = {c: CITY_DIRS[c] for c in args.cities if c in CITY_DIRS}

    # Phase 1: Build full index
    type_to_samples = build_full_index(cities)

    # Load per-type budget if provided (overrides args.cap on a per-type basis)
    per_type_budget = None
    if args.per_type_budget:
        with open(args.per_type_budget) as f:
            per_type_budget = json.load(f)
        print(f"\n[Budget] Loaded per-type budget from {args.per_type_budget}: "
              f"{len(per_type_budget)} types, sum={sum(per_type_budget.values()):,}")

    def type_cap(stype, count):
        if per_type_budget is not None:
            return min(count, per_type_budget.get(stype, 0))
        return min(count, args.cap)

    # Show type distribution
    type_counts = {t: len(samples) for t, samples in type_to_samples.items()}
    sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])

    print(f"\n{'Type':<55} {'Available':>10} {'Keep':>8}")
    print("-" * 77)
    total_keep = 0
    for stype, count in sorted_types:
        keep = type_cap(stype, count)
        total_keep += keep
        if per_type_budget is not None:
            budgeted = per_type_budget.get(stype, 0)
            status = f"budget={budgeted}" if budgeted > 0 else "skip"
        else:
            status = "capped" if count > args.cap else "all"
        print(f"  {stype:<53} {count:>10,} {keep:>8,}  {status}")

    if per_type_budget is None:
        types_at_cap = sum(1 for _, c in sorted_types if c >= args.cap)
        types_below = len(sorted_types) - types_at_cap
        print(f"\n[Phase 1] {len(sorted_types)} types, {types_at_cap} at cap, "
              f"{types_below} below cap")
    else:
        print(f"\n[Phase 1] {len(sorted_types)} types indexed, "
              f"using per-type budget")
    print(f"[Phase 1] Will extract: {total_keep:,} samples")

    if args.dry_run:
        print("\n[Dry run] Stopping here.")
        return

    # Phase 2: Sample per type, then split by city
    label = "per-type budget" if per_type_budget is not None else f"{args.cap}/type"
    print(f"\n[Phase 2] Sampling {label} (seed={args.seed})...")
    city_pairs = defaultdict(list)      # city → [(db_path, token)]
    city_types = defaultdict(list)      # city → [scenario_type]

    for stype, samples in type_to_samples.items():
        n_keep = type_cap(stype, len(samples))
        if n_keep <= 0:
            continue
        if n_keep < len(samples):
            chosen = random.sample(samples, n_keep)
        else:
            chosen = samples
        for city, db_path, token in chosen:
            city_pairs[city].append((db_path, token))
            city_types[city].append(stype)

    for city in cities:
        print(f"  {city}: {len(city_pairs[city]):,} selected")
    print(f"  Total: {sum(len(v) for v in city_pairs.values()):,}")

    # Extract each city sequentially (single-disk I/O per city)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    all_type_buffers = []
    total_extracted = 0

    for city in cities:
        if not city_pairs[city]:
            continue

        data, type_buffer = extract_city(
            city, city_pairs[city], city_types[city], args
        )
        if data is None:
            continue

        # Save per-city cache
        cache_path = os.path.join(cfg.CHECKPOINT_DIR,
                                   f"stage_cache_{city}_balanced.pt")
        print(f"[Save] {city} → {cache_path}")
        torch.save(data, cache_path)
        size_gb = os.path.getsize(cache_path) / 1e9
        print(f"[Save] {data['n_samples']:,} samples, {size_gb:.2f} GB")

        # Save type sidecar
        type_path = os.path.join(cfg.CHECKPOINT_DIR,
                                  f"stage_cache_{city}_balanced_types.pt")
        city_counts = Counter(type_buffer)
        torch.save({
            'scenario_types': type_buffer,
            'n_cached': data['n_samples'],
            'n_indexed': len(city_pairs[city]),
            'type_counts': dict(city_counts),
        }, type_path)

        total_extracted += data['n_samples']
        all_type_buffers.extend(type_buffer)

        # Sanity
        print(f"[{city}] Shapes:")
        for k in ['agents_history', 'agents_mask', 'agents_seq', 'agents_now',
                   'gt_trajectory', 'mode_label', 'map_lanes', 'map_lanes_mask',
                   'map_polygons', 'map_polygons_mask']:
            print(f"  {k}: {tuple(data[k].shape)}")
        print(f"  Avg valid agents:   {data['agents_mask'].sum(1).mean():.1f}")
        print(f"  Avg valid lanes:    {data['map_lanes_mask'].sum(1).mean():.1f}")
        print(f"  Avg valid polygons: {data['map_polygons_mask'].sum(1).mean():.1f}")
        has_nan = any(torch.isnan(data[k]).any() for k in
                      ['agents_history', 'agents_seq', 'gt_trajectory', 'map_lanes', 'map_polygons'])
        print(f"  NaN check: {'FAIL' if has_nan else 'OK'}")

        del data

    # Summary
    final_counts = Counter(all_type_buffers)
    print(f"\n[Done] Total extracted: {total_extracted:,} samples across {len(final_counts)} types")
    print(f"\nFinal type distribution:")
    for stype, count in final_counts.most_common():
        print(f"  {stype:<55} {count:>6,}")
    print(f"\nPer-city caches saved. Merge with: python merge_balanced.py")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Balanced extraction (PDM protocol)")
    p.add_argument('--cap', type=int, default=4000,
                   help='Max scenarios per type (default: 4000, matching paper)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--cities', nargs='+', default=None,
                   choices=['train_boston', 'train_pittsburgh', 'train_singapore', 'train_vegas'],
                   help='Cities to include (default: all 3)')
    p.add_argument('--per_type_budget', type=str, default=None,
                   help='Path to JSON {type: n} overriding --cap per-type (missing types = skipped)')
    p.add_argument('--dry_run', action='store_true',
                   help='Phase 1 only — show type counts, no extraction')
    args = p.parse_args()
    extract(args)
