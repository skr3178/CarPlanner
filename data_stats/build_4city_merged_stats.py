"""Build per-type stats for the 4-city merged paper-balanced cache.

Uses the per-city _types.pt sidecars (one cached scenario = 1 unit; sidecars carry
post-extraction type_counts) and applies the global per-type 4000-cap merge rule
that produced stage_cache_train_4city_paper_balanced.pt.

Outputs:
  data_stats/four_city_merged_stats.txt
  data_stats/four_city_merged_stats.csv
"""
import csv
import os
import torch
from collections import Counter

CITIES = ['boston', 'vegas', 'pittsburgh', 'singapore']
SIDECAR_FMT = 'checkpoints/stage_cache_train_{}_balanced_types.pt'
CAP = 4000

city_tc = {}
for c in CITIES:
    p = SIDECAR_FMT.format(c)
    if not os.path.exists(p):
        print(f'[WARN] missing {p} — skipping {c}')
        continue
    d = torch.load(p, map_location='cpu', weights_only=False)
    city_tc[c] = (d['n_cached'], d['type_counts'])

pool = Counter()
for c, (_, tc) in city_tc.items():
    for t, ct in tc.items():
        pool[t] += ct

post_cap = {t: min(c, CAP) for t, c in pool.items()}

sorted_items = sorted(post_cap.items(), key=lambda x: -x[1])
total_after = sum(post_cap.values())
total_before = sum(pool.values())

lines = []
lines.append('# Source: 4-city paper-balanced merge')
lines.append('# Per-city sidecars (one cached scenario per row, post-extraction):')
for c, (n, tc) in city_tc.items():
    lines.append(f'#   {c:12s} n_cached={n:>7,}  unique_types={len(tc)}')
lines.append(f'# Pool unique types: {len(pool)}')
lines.append(f'# Pool sum (pre-cap): {total_before:,}')
lines.append(f'# Cap: per-type {CAP} (global)')
lines.append(f'# After cap: {total_after:,} samples (matches stage_cache_train_4city_paper_balanced.pt)')
lines.append('')
lines.append(f'  #   {"Scenario type":56s} {"BOS":>6s} {"VEG":>6s} {"PIT":>6s} {"SNG":>6s} '
             f'{"pool":>7s} {"capped":>7s}')
lines.append('  ' + '-' * 100)
for i, (t, c_after) in enumerate(sorted_items, 1):
    b = city_tc.get('boston',     (0, {}))[1].get(t, 0)
    v = city_tc.get('vegas',      (0, {}))[1].get(t, 0)
    pp = city_tc.get('pittsburgh',(0, {}))[1].get(t, 0)
    s = city_tc.get('singapore',  (0, {}))[1].get(t, 0)
    lines.append(f'  {i:3d}  {t:56s} {b:>6,} {v:>6,} {pp:>6,} {s:>6,} {pool[t]:>7,} {c_after:>7,}')
lines.append('  ' + '-' * 100)
lines.append(f'  {len(post_cap)} types {"":50s} '
             f'{sum(city_tc["boston"][1].values()):>6,} {sum(city_tc["vegas"][1].values()):>6,} '
             f'{sum(city_tc["pittsburgh"][1].values()):>6,} {sum(city_tc["singapore"][1].values()):>6,} '
             f'{total_before:>7,} {total_after:>7,}')

report = '\n'.join(lines)
print(report)

OUT_TXT = 'data_stats/four_city_merged_stats.txt'
OUT_CSV = 'data_stats/four_city_merged_stats.csv'
with open(OUT_TXT, 'w') as f:
    f.write(report + '\n')
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['rank', 'scenario_type', 'boston', 'vegas', 'pittsburgh', 'singapore',
                'pool', 'after_cap_4000'])
    for i, (t, c_after) in enumerate(sorted_items, 1):
        b = city_tc.get('boston',    (0, {}))[1].get(t, 0)
        v = city_tc.get('vegas',     (0, {}))[1].get(t, 0)
        pp = city_tc.get('pittsburgh',(0, {}))[1].get(t, 0)
        s = city_tc.get('singapore', (0, {}))[1].get(t, 0)
        w.writerow([i, t, b, v, pp, s, pool[t], c_after])
print(f'\n[Saved] {OUT_TXT}')
print(f'[Saved] {OUT_CSV}')
