"""Count scenario types for a list of scenario_tokens (e.g. val14, test14_random YAMLs).

Joins YAML.scenario_tokens (hex strings) against scenario_tag.token (BLOB) across
all .db files in <db_dir>. Each scenario_token resolves to one type via
nuPlan's scenario_tag schema.

Usage:
    python count_scenario_types_yaml.py <yaml_path> <db_dir> [--out ...] [--csv ...]
"""
import argparse
import csv
import os
import sqlite3
import sys
import time
import yaml
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob


def lookup_db(path, token_blobs):
    """Return list of (lidar_pc_token_hex, type) for tags whose scenario start
    (lidar_pc_token) is in the requested token set. A scenario can carry
    multiple type tags so this can return >1 row per token."""
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    cur = con.cursor()
    out = []
    CHUNK = 500
    blobs = list(token_blobs)
    for i in range(0, len(blobs), CHUNK):
        chunk = blobs[i:i+CHUNK]
        q = (f"SELECT lidar_pc_token, type FROM scenario_tag "
             f"WHERE lidar_pc_token IN ({','.join('?'*len(chunk))})")
        for tok, typ in cur.execute(q, chunk).fetchall():
            out.append((tok.hex(), typ))
    con.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('yaml_path')
    p.add_argument('db_dir')
    p.add_argument('--out', default=None)
    p.add_argument('--csv', default=None)
    p.add_argument('--workers', type=int, default=24)
    args = p.parse_args()

    with open(args.yaml_path) as f:
        cfg = yaml.safe_load(f)
    tokens_hex = cfg.get('scenario_tokens') or []
    if not tokens_hex:
        print(f'[ERR] no scenario_tokens in {args.yaml_path}', file=sys.stderr)
        sys.exit(1)
    print(f'[YAML] {args.yaml_path}: {len(tokens_hex)} scenario_tokens', flush=True)

    token_blobs = [bytes.fromhex(t) for t in tokens_hex]

    db_files = sorted(glob(os.path.join(args.db_dir, '*.db')))
    print(f'[DBs] {args.db_dir}: {len(db_files)} files', flush=True)

    rows = []           # (token_hex, type) with possible duplicate tokens
    resolved = set()    # set of token_hex with >=1 tag
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(lookup_db, f, token_blobs): f for f in db_files}
        for fut in as_completed(futs):
            try:
                for tok_hex, typ in fut.result():
                    rows.append((tok_hex, typ))
                    resolved.add(tok_hex)
            except Exception as e:
                print(f'ERR {futs[fut]}: {e}', flush=True)
            done += 1
            if done % 200 == 0:
                print(f'  [{done}/{len(db_files)}] tags={len(rows)} resolved_tokens={len(resolved)}/'
                      f'{len(tokens_hex)} elapsed={time.time()-t0:.1f}s', flush=True)
    elapsed = time.time() - t0
    print(f'[Done] {len(db_files)} dbs in {elapsed:.1f}s — '
          f'tags={len(rows)} resolved_tokens={len(resolved)}/{len(tokens_hex)}', flush=True)
    missing = set(tokens_hex) - resolved
    if missing:
        print(f'[WARN] {len(missing)} tokens unresolved (first 5: {list(missing)[:5]})', flush=True)

    counts = Counter(typ for _, typ in rows)
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])

    lines = []
    lines.append(f'# Source: {args.yaml_path}')
    lines.append(f'# Joined against: {args.db_dir} ({len(db_files)} .db files)')
    lines.append(f'# Tokens in YAML: {len(tokens_hex)}    Resolved: {len(resolved)}    Missing: {len(missing)}')
    lines.append(f'# Total tags (a scenario can carry multiple): {len(rows)}    Unique types: {len(counts)}')
    lines.append('')
    lines.append(f'  #   {"Scenario type":56s} {"Tag count":>12s}')
    lines.append('  ' + '-' * 76)
    for i, (t, c) in enumerate(sorted_items, 1):
        lines.append(f'  {i:3d}  {t:56s} {c:>12,}')
    lines.append('  ' + '-' * 76)
    lines.append(f'  {len(counts)} types {"":50s} Total: {sum(counts.values()):>12,}')
    report = '\n'.join(lines)
    print()
    print(report)

    if args.out:
        with open(args.out, 'w') as f:
            f.write(report + '\n')
        print(f'\n[Saved] {args.out}')

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['rank', 'scenario_type', 'tag_count'])
            for i, (t, c) in enumerate(sorted_items, 1):
                w.writerow([i, t, c])
        print(f'[Saved] {args.csv}')


if __name__ == '__main__':
    main()
