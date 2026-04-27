"""Post-hoc CLS-NR aggregator from nuPlan per-scenario metric parquets.

The nuPlan metric_aggregator skips when the output_dir doesn't carry the
challenge name, but the per-scenario metric files are still written. This
script reads them and applies the closed_loop_nonreactive_agents_weighted_average
formula directly:

  CLS-NR_scenario = ∏(multipliers) × Σ(w·score)/Σ(w) × 100
  CLS-NR_aggregate = mean over scenarios

Multipliers      (0, 0.5, or 1): no_ego_at_fault_collisions,
                                  drivable_area_compliance,
                                  ego_is_making_progress,
                                  driving_direction_compliance
Weighted (∈[0,1]): ego_progress_along_expert_route (w=5),
                   time_to_collision_within_bound (w=5),
                   speed_limit_compliance (w=4),
                   ego_is_comfortable (w=2)
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import pandas as pd

MULTIPLIERS = [
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "ego_is_making_progress",
    "driving_direction_compliance",
]
WEIGHTED = {
    "ego_progress_along_expert_route": 5.0,
    "time_to_collision_within_bound": 5.0,
    "speed_limit_compliance": 4.0,
    "ego_is_comfortable": 2.0,
}


def find_metrics_dir(run_dir):
    cands = glob.glob(os.path.join(run_dir, "**/metrics"), recursive=True)
    cands = [c for c in cands if os.path.isdir(c)]
    if not cands:
        raise FileNotFoundError(f"No metrics/ subdir under {run_dir}")
    return sorted(cands, key=len)[0]


def read_metric(mdir, name):
    p = os.path.join(mdir, f"{name}.parquet")
    if not os.path.isfile(p):
        return None
    return pd.read_parquet(p)


def aggregate(run_dir):
    mdir = find_metrics_dir(run_dir)
    print(f"[Aggregate] metrics dir: {mdir}")

    # Per-scenario score: dict[scenario_name] -> {metric: score}
    scenario_scores = defaultdict(dict)
    scenario_type   = {}

    for m in MULTIPLIERS + list(WEIGHTED):
        df = read_metric(mdir, m)
        if df is None:
            print(f"  [warn] metric file missing: {m}.parquet")
            continue
        for _, row in df.iterrows():
            sn = row["scenario_name"]
            scenario_scores[sn][m] = float(row["metric_score"])
            scenario_type[sn] = row["scenario_type"]

    # Per-scenario CLS-NR
    rows = []
    for sn, ms in scenario_scores.items():
        mult = 1.0
        for k in MULTIPLIERS:
            mult *= ms.get(k, 1.0)
        wsum, wnorm = 0.0, 0.0
        for k, w in WEIGHTED.items():
            if k in ms:
                wsum += w * ms[k]
                wnorm += w
        weighted_avg = wsum / wnorm if wnorm > 0 else 0.0
        rows.append({
            "scenario_name": sn,
            "scenario_type": scenario_type[sn],
            "multiplier_prod": mult,
            "weighted_avg":   weighted_avg,
            "cls_nr":         mult * weighted_avg * 100.0,
            **{f"mult_{k}": ms.get(k) for k in MULTIPLIERS},
            **{f"score_{k}": ms.get(k) for k in WEIGHTED},
        })
    df = pd.DataFrame(rows)
    return df


def report(df, name):
    n = len(df)
    cls_nr = df["cls_nr"].mean()
    print(f"\n=== {name} ({n} scenarios) ===")
    print(f"  CLS-NR (composite):       {cls_nr:.2f}")
    print(f"  Per-multiplier pass rate (% of scenarios where score == 1):")
    for k in MULTIPLIERS:
        col = f"mult_{k}"
        if col in df.columns:
            pct = (df[col] >= 0.999).mean() * 100
            avg = df[col].mean()
            print(f"    {k:<35} {avg*100:5.1f}% avg  ({pct:5.1f}% perfect)")
    print(f"  Per-score-metric mean (∈[0,1]):")
    for k in WEIGHTED:
        col = f"score_{k}"
        if col in df.columns:
            print(f"    {k:<35} {df[col].mean():.3f}  (weight {WEIGHTED[k]})")
    print(f"  Per-type CLS-NR:")
    type_cls = df.groupby("scenario_type")["cls_nr"].agg(["mean", "count"])
    for t, row in type_cls.sort_values("mean").iterrows():
        print(f"    {t:<58} {row['mean']:6.2f}  (n={int(row['count'])})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+",
                   help="One or more nuplan_eval/<run>/ directories")
    args = p.parse_args()
    for rd in args.run_dirs:
        df = aggregate(rd)
        report(df, os.path.basename(rd.rstrip("/")))


if __name__ == "__main__":
    main()
