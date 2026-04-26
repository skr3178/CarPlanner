"""
Build a 4-city balanced training cache that matches the paper's literal recipe:
"4,000 scenarios per type" applied GLOBALLY across all 4 cities (not per-city).

Algorithm:
  1. Load each city's cache (samples) and its type sidecar (per-sample scenario type tag).
  2. Pool sample indices across all 4 cities, grouped by scenario type.
  3. For each type, randomly select min(N, 4000) samples.
  4. Concatenate the selected samples into one cache and save.

Output: checkpoints/stage_cache_train_4city_paper_balanced.pt
"""
import os
import random
import time
from collections import defaultdict, Counter

import torch

CITIES = ['boston', 'vegas', 'pittsburgh', 'singapore']
CAP_PER_TYPE = 4000
TARGET_TOTAL = 176_218   # paper Section 4.1 — exact training-set size
SEED = 42
OUT_PATH = 'checkpoints/stage_cache_train_4city_paper_balanced.pt'

random.seed(SEED)


def main():
    t0 = time.time()
    caches = {}
    types_per_city = {}

    print('[Load] per-city caches and type sidecars')
    for city in CITIES:
        cache_path = f'checkpoints/stage_cache_train_{city}_balanced.pt'
        types_path = f'checkpoints/stage_cache_train_{city}_balanced_types.pt'
        d = torch.load(cache_path, map_location='cpu')
        sd = torch.load(types_path, map_location='cpu')
        caches[city] = d
        types_per_city[city] = sd['scenario_types']
        n = d['n_samples']
        nt = len(sd['scenario_types'])
        assert n == nt, f'{city}: {n} samples vs {nt} type tags — mismatch'
        print(f'  {city:12s}  {n:>7,} samples  {len(set(sd["scenario_types"]))} unique types')

    # Build a global pool of (city, sample_idx) keyed by scenario type
    print('\n[Pool] grouping samples by scenario type across all cities...')
    pool = defaultdict(list)
    for city in CITIES:
        for i, t in enumerate(types_per_city[city]):
            pool[t].append((city, i))

    print(f'  total types in pool: {len(pool)}')
    print(f'  type counts before cap (top 10):')
    for t, lst in sorted(pool.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f'    {t:55s}  {len(lst):>6,}')

    # Apply global per-type cap
    print(f'\n[Cap] global cap = {CAP_PER_TYPE} per type')
    selected = []  # list of (city, sample_idx)
    capped, kept_all = 0, 0
    for t, lst in pool.items():
        if len(lst) > CAP_PER_TYPE:
            chosen = random.sample(lst, CAP_PER_TYPE)
            capped += 1
        else:
            chosen = lst
            kept_all += 1
        selected.extend(chosen)
    print(f'  {capped} types capped at {CAP_PER_TYPE}, {kept_all} types kept all (≤cap)')
    print(f'  after per-type cap: {len(selected):,} samples')

    # Optional final downsample to match paper's exact total (176,218)
    if TARGET_TOTAL is not None:
        if len(selected) > TARGET_TOTAL:
            selected = random.sample(selected, TARGET_TOTAL)
            print(f'  downsampled to paper target: {len(selected):,} samples')
        elif len(selected) < TARGET_TOTAL:
            print(f'  WARNING: only {len(selected):,} available, paper target {TARGET_TOTAL:,} unreachable')
        else:
            print(f'  already at paper target')

    # City breakdown of the final selection
    city_breakdown = Counter(c for c, _ in selected)
    print(f'\n[Composition] final mix:')
    n_total = len(selected)
    for c in CITIES:
        cnt = city_breakdown.get(c, 0)
        print(f'  {c:12s}  {cnt:>7,}  ({100*cnt/n_total:5.2f}%)')

    # Group selected indices per city for batched indexing
    indices_per_city = defaultdict(list)
    for c, i in selected:
        indices_per_city[c].append(i)
    for c in indices_per_city:
        indices_per_city[c] = torch.tensor(indices_per_city[c], dtype=torch.long)

    # Tensor keys to concatenate (everything that has first-dim = n_samples)
    sample = caches[CITIES[0]]
    tensor_keys = [k for k, v in sample.items()
                   if torch.is_tensor(v) and v.shape[0] == sample['n_samples']]
    print(f'\n[Merge] concatenating {len(tensor_keys)} tensor keys...')

    merged = {}
    for k in tensor_keys:
        parts = []
        for c in CITIES:
            idx = indices_per_city.get(c)
            if idx is None or len(idx) == 0:
                continue
            parts.append(caches[c][k][idx])
        merged[k] = torch.cat(parts, dim=0)
        print(f'  {k:24s} → {tuple(merged[k].shape)}')

    merged['n_samples'] = sum(len(idx) for idx in indices_per_city.values())
    merged['split'] = 'train_4city_paper_balanced'
    # Per-sample city tag, in same row order as merged tensors
    merged['city'] = []
    for c in CITIES:
        merged['city'].extend([c] * len(indices_per_city.get(c, [])))

    print(f'\n[Save] {OUT_PATH}  total n_samples={merged["n_samples"]:,}')
    torch.save(merged, OUT_PATH)
    size_gb = os.path.getsize(OUT_PATH) / 1e9
    print(f'[Save] {size_gb:.2f} GB')

    # Sanity: lat-bin distribution on the merged cache
    import config as cfg
    labels = merged['mode_label'].tolist()
    n = len(labels)
    lat = Counter(int(m) % cfg.N_LAT for m in labels)
    print(f'\n[Verify] Combined lateral distribution:')
    for la in range(cfg.N_LAT):
        c = lat.get(la, 0)
        bar = '█' * int(60 * c / n)
        print(f'  lat={la}  {100*c/n:5.2f}%  {bar}')

    print(f'\nDone in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
