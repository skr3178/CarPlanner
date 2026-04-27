"""Tabulate scenario_tag.type counts across all .db files in a nuPlan shard directory.

Usage:
    python count_scenario_types.py <db_dir> [--out <txt>] [--csv <csv>] [--workers N]

Example:
    python count_scenario_types.py data/cache/train_vegas_5 \
        --out data_stats/vegas_5_raw_stats.txt \
        --csv data_stats/vegas_5_raw_stats.csv
"""
import argparse
import csv
import os
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob


def count_db(path):
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    cur = con.cursor()
    rows = cur.execute('SELECT type, COUNT(*) FROM scenario_tag GROUP BY type').fetchall()
    con.close()
    return Counter(dict(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('db_dir', help='directory containing nuPlan .db files')
    p.add_argument('--out', default=None, help='output text report path')
    p.add_argument('--csv', default=None, help='output csv path')
    p.add_argument('--workers', type=int, default=24)
    args = p.parse_args()

    db_files = sorted(glob(os.path.join(args.db_dir, '*.db')))
    if not db_files:
        print(f'[ERR] no .db files in {args.db_dir}', file=sys.stderr)
        sys.exit(1)
    print(f'[{os.path.basename(args.db_dir)}] {len(db_files)} db files', flush=True)

    t0 = time.time()
    total = Counter()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(count_db, p): p for p in db_files}
        for f in as_completed(futs):
            try:
                total.update(f.result())
            except Exception as e:
                print(f'ERR {futs[f]}: {e}', flush=True)
            done += 1
            if done % 50 == 0:
                print(f'  [{done}/{len(db_files)}] elapsed={time.time()-t0:.1f}s', flush=True)
    elapsed = time.time() - t0
    n_tags = sum(total.values())
    print(f'[Done] {len(db_files)} dbs in {elapsed:.1f}s', flush=True)
    print(f'[Total] unique types={len(total)}  total tags={n_tags:,}', flush=True)

    sorted_items = sorted(total.items(), key=lambda x: -x[1])

    lines = []
    lines.append(f'# Source: {args.db_dir}')
    lines.append(f'# .db files: {len(db_files)}    queried in {elapsed:.1f}s on {args.workers} threads')
    lines.append(f'# Unique types: {len(total)}    Total scenario_tag rows: {n_tags:,}')
    lines.append('')
    lines.append(f'  #   {"Scenario type":56s} {"Tag count":>12s}')
    lines.append('  ' + '-' * 76)
    for i, (t, c) in enumerate(sorted_items, 1):
        lines.append(f'  {i:3d}  {t:56s} {c:>12,}')
    lines.append('  ' + '-' * 76)
    lines.append(f'  {len(total)} types {"":50s} Total: {n_tags:>12,}')
    report = '\n'.join(lines)
    print()
    print(report)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(report + '\n')
        print(f'\n[Saved] {args.out}')

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['rank', 'scenario_type', 'tag_count'])
            for i, (t, c) in enumerate(sorted_items, 1):
                w.writerow([i, t, c])
        print(f'[Saved] {args.csv}')


if __name__ == '__main__':
    main()
