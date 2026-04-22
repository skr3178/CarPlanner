"""
Merge per-city balanced caches into a single training dataset.

For 3 cities (no Vegas): keeps ALL samples (~117K), no further capping needed.
For 4 cities (with Vegas): caps to 4000/type, trims to 176K target.

Usage:
    python merge_balanced.py                    # 3-city, keep all
    python merge_balanced.py --target 176218    # 4-city, cap to paper target
    python merge_balanced.py --dry_run
"""

import os
import sys
import argparse
import random
from collections import Counter, defaultdict

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import torch
import config as cfg


ALL_CITIES = ['train_boston', 'train_pittsburgh', 'train_singapore']


def load_type_index(split):
    path = os.path.join(cfg.CHECKPOINT_DIR, f"stage_cache_{split}_balanced_types.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Type index not found: {path}\n"
                                f"Run: python extract_balanced.py")
    return torch.load(path, map_location='cpu')


def main(args):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)

    # Discover available cities
    cities = []
    for city in ALL_CITIES + (args.extra_cities or []):
        type_path = os.path.join(cfg.CHECKPOINT_DIR, f"stage_cache_{city}_balanced_types.pt")
        cache_path = os.path.join(cfg.CHECKPOINT_DIR, f"stage_cache_{city}_balanced.pt")
        if os.path.isfile(type_path) and os.path.isfile(cache_path):
            cities.append(city)
        else:
            print(f"  [skip] {city} — cache not found")

    if not cities:
        print("No city caches found. Run extract_balanced.py first.")
        return

    # Step 1: Load type indexes for all cities
    print(f"[Merge] Loading type indexes for {len(cities)} cities...")

    city_type_lists = {}
    city_n_cached = {}
    combined_types = Counter()

    for city in cities:
        info = load_type_index(city)
        types = info['scenario_types']
        n_cached = info['n_cached']
        n_indexed = info['n_indexed']
        city_type_lists[city] = types[:n_cached]
        city_n_cached[city] = n_cached
        city_counts = Counter(types[:n_cached])
        combined_types += city_counts
        print(f"  {city}: {n_cached:,} samples, {len(city_counts)} types "
              f"(drop rate: {(n_indexed-n_cached)/n_indexed*100:.1f}%)")

    total_available = sum(city_n_cached.values())
    n_types = len(combined_types)
    print(f"\n[Merge] Total available: {total_available:,} samples, {n_types} types")

    # Determine target and cap
    target = args.target if args.target else total_available
    sorted_types = combined_types.most_common()

    if target >= total_available:
        cap = max(combined_types.values())
        actual_total = total_available
        print(f"\n[Merge] Keeping ALL {total_available:,} samples (target={target:,})")
    else:
        def compute_total(c):
            return sum(min(count, c) for _, count in sorted_types)

        lo, hi = 1, max(combined_types.values())
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if compute_total(mid) <= target:
                lo = mid
            else:
                hi = mid - 1
        cap = lo
        actual_total = compute_total(cap)

        if actual_total < target:
            cap_plus = cap + 1
            total_plus = compute_total(cap_plus)
            if total_plus >= target:
                cap = cap_plus
                actual_total = total_plus

        print(f"\n[Merge] Computed per-type cap: {cap:,}")
        print(f"[Merge] Projected total: {actual_total:,} (target: {target:,})")

    # Show per-type plan
    types_at_cap = 0
    types_below_cap = 0
    edge_case_total = 0

    print(f"\n{'Type':<50} {'Available':>10} {'Keep':>8} {'Status':>10}")
    print("-" * 82)
    for stype, count in sorted_types:
        keep = min(count, cap)
        if count >= cap:
            types_at_cap += 1
            status = "capped"
        else:
            types_below_cap += 1
            edge_case_total += keep
            status = "keep all"
        print(f"  {stype:<48} {count:>10,} {keep:>8,} {status:>10}")

    print(f"\n[Merge] Types at cap ({cap}): {types_at_cap}")
    print(f"[Merge] Types below cap (keep all): {types_below_cap} "
          f"({edge_case_total:,} samples)")
    print(f"[Merge] Total to merge: {actual_total:,}")

    if args.dry_run:
        print("\n[Merge] Dry run — not building cache.")
        return

    # Step 3: Build per-type sample indices across all cities
    print(f"\n[Merge] Building per-type sample lists...")
    type_to_samples = defaultdict(list)

    for city in cities:
        types = city_type_lists[city]
        for local_idx, stype in enumerate(types):
            type_to_samples[stype].append((city, local_idx))

    # Step 4: Subsample per type
    print(f"[Merge] Subsampling per type (cap={cap}, seed={seed})...")
    selected = defaultdict(list)

    for stype, samples in type_to_samples.items():
        n_keep = min(len(samples), cap)
        if n_keep < len(samples):
            chosen = random.sample(samples, n_keep)
        else:
            chosen = samples
        for city, local_idx in chosen:
            selected[city].append(local_idx)

    for city in cities:
        selected[city].sort()
        print(f"  {city}: {len(selected[city]):,} samples selected")

    total_selected = sum(len(v) for v in selected.values())
    print(f"  Total selected: {total_selected:,}")

    if total_selected > target:
        excess = total_selected - target
        print(f"  Trimming {excess} excess samples randomly...")
        all_selected = [(city, idx) for city in cities for idx in selected[city]]
        random.shuffle(all_selected)
        remove_set = set(all_selected[:excess])
        for city in cities:
            selected[city] = [idx for idx in selected[city]
                              if (city, idx) not in remove_set]
        total_selected = sum(len(v) for v in selected.values())
        print(f"  After trim: {total_selected:,}")

    # Step 5: Load caches and build merged dataset
    print(f"\n[Merge] Loading city caches and extracting selected samples...")
    tensor_keys = ['agents_history', 'agents_mask', 'agents_seq', 'agents_now',
                   'gt_trajectory', 'mode_label', 'map_lanes', 'map_lanes_mask',
                   'map_polygons', 'map_polygons_mask']

    merged_buffers = {k: [] for k in tensor_keys}
    merged_types = []

    for city in cities:
        cache_path = os.path.join(cfg.CHECKPOINT_DIR, f"stage_cache_{city}_balanced.pt")
        print(f"  Loading {city}...")
        data = torch.load(cache_path, map_location='cpu')

        indices = selected[city]
        idx_tensor = torch.tensor(indices, dtype=torch.long)

        for k in tensor_keys:
            merged_buffers[k].append(data[k][idx_tensor])

        types = city_type_lists[city]
        for idx in indices:
            merged_types.append(types[idx])

        del data
        print(f"    Selected {len(indices):,} / {city_n_cached[city]:,}")

    # Concatenate
    print(f"\n[Merge] Concatenating...")
    merged = {}
    for k in tensor_keys:
        merged[k] = torch.cat(merged_buffers[k], dim=0)
        print(f"  {k}: {tuple(merged[k].shape)}")
    merged['n_samples'] = total_selected
    merged['split'] = 'train_all'

    # Save
    out_path = os.path.join(cfg.CHECKPOINT_DIR, f"stage_cache_{args.output}.pt")
    print(f"\n[Merge] Saving to {out_path}...")
    torch.save(merged, out_path)

    size_gb = os.path.getsize(out_path) / 1e9
    print(f"[Merge] Done. {total_selected:,} samples, {size_gb:.2f} GB")

    # Final type distribution
    final_counts = Counter(merged_types)
    print(f"\n[Merge] Final type distribution ({len(final_counts)} types):")
    print(f"{'Type':<50} {'Count':>8}")
    print("-" * 62)
    for stype, count in final_counts.most_common():
        print(f"  {stype:<48} {count:>8,}")

    # Save type sidecar for the merged cache
    type_path = os.path.join(cfg.CHECKPOINT_DIR, f"stage_cache_{args.output}_types.pt")
    torch.save({
        'scenario_types': merged_types,
        'n_cached': total_selected,
        'type_counts': dict(final_counts),
    }, type_path)
    print(f"[Merge] Type index saved to {type_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Build balanced merged cache")
    p.add_argument('--target', type=int, default=None,
                   help='Target dataset size (default: keep all available)')
    p.add_argument('--output', default='train_all_balanced',
                   help='Output cache name (default: train_all_balanced)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility')
    p.add_argument('--extra_cities', nargs='+', default=None,
                   help='Additional city names beyond the default 3')
    p.add_argument('--dry_run', action='store_true',
                   help='Show plan without building cache')
    args = p.parse_args()
    main(args)
