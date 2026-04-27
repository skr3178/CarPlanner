"""
CarPlanner — open-loop eval infrastructure sanity check.

Mirrors what `log_future_planner` did for closed-loop (CLS-NR=93.08): feeds
ground-truth trajectories through the same metric functions used by
`eval_stage_b.py` and reports per-metric expected-vs-actual values.

Anything close to its expected ceiling means the metric code, frame
transforms, and cache contents are correct. The remaining gap during real
Stage-B eval is then a model issue, not infrastructure.

Usage:
    python eval_sanity_open_loop.py
    python eval_sanity_open_loop.py --split reduced_val14
    python eval_sanity_open_loop.py --n_eval 200
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import config as cfg
from data_loader import PreextractedDataset
from eval_stage_b import (
    _compute_consistent_ratio,
    _compute_ol_col_area,
    SPLIT_TO_CACHE,
)


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Sanity OL] Device: {device}")
    print(f"[Sanity OL] Cache: {args.cache}")
    dataset = PreextractedDataset(args.cache, device=device)
    n_total = len(dataset)
    n_eval = min(args.n_eval, n_total) if args.n_eval > 0 else n_total
    print(f"[Sanity OL] {n_eval}/{n_total} samples\n")

    indices = np.linspace(0, n_total - 1, n_eval, dtype=int).tolist()

    stats = {
        'ade': [], 'fde': [], 'l_gen': [], 'l_sel': [],
        'mode_correct': [], 'top5_correct': [], 'median_rank_input': [],
        'consistent_lat': [], 'consistent_lon': [],
        'col_mean': [], 'col_min': [], 'col_max': [],
        'area_mean': [], 'area_min': [], 'area_max': [],
    }

    t0 = time.time()
    for done, idx in enumerate(indices, 1):
        sample = dataset[idx]
        gt = sample['gt_trajectory'].cpu().numpy()                 # (T, 3)
        mode_label = int(sample['mode_label'].item())

        # ── GT pass-through ────────────────────────────────────────────────
        pred = gt.copy()                                           # pred == gt
        # 60 candidate trajectories: all equal to gt (model bypass)
        all_trajs_np = np.broadcast_to(gt, (cfg.N_MODES,) + gt.shape).copy()
        # mode_logits: one-hot at mode_label so chosen_mode == mode_label,
        # CE loss = 0 with infinitely confident logits
        logits = torch.full((cfg.N_MODES,), -1e4)
        logits[mode_label] = 1e4
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        # ── Metric copies — same code path as eval_stage_b.py ──────────────
        diff = pred[:, :2] - gt[:, :2]
        stats['ade'].append(float(np.mean(np.linalg.norm(diff, axis=1))))
        stats['fde'].append(float(np.linalg.norm(pred[-1, :2] - gt[-1, :2])))
        stats['l_gen'].append(float(np.sum(np.abs(pred - gt))))
        stats['l_sel'].append(float(-log_probs[mode_label]))

        order = torch.argsort(probs, descending=True)
        gt_rank = int((order == mode_label).nonzero(as_tuple=True)[0].item()) + 1
        chosen_mode = int(torch.argmax(probs).item())
        stats['mode_correct'].append(chosen_mode == mode_label)
        stats['top5_correct'].append(gt_rank <= 5)
        stats['median_rank_input'].append(gt_rank)

        lat_r, lon_r = _compute_consistent_ratio(all_trajs_np, gt)
        stats['consistent_lat'].append(lat_r)
        stats['consistent_lon'].append(lon_r)

        agents_now_np  = sample['agents_now'].cpu().numpy()
        agents_mask_np = sample['agents_history_mask'].cpu().numpy()
        ml_np  = sample['map_lanes'].cpu().numpy()
        mlm_np = sample['map_lanes_mask'].cpu().numpy()
        rp_np  = sample['route_polylines'].cpu().numpy() if 'route_polylines' in sample else None
        rm_np  = sample['route_mask'].cpu().numpy() if 'route_mask' in sample else None
        cm, cmin, cmax, am, amin, amax = _compute_ol_col_area(
            all_trajs_np, agents_now_np, agents_mask_np, ml_np, mlm_np,
            route_polylines_np=rp_np, route_mask_np=rm_np)
        stats['col_mean'].append(cm);  stats['col_min'].append(cmin);  stats['col_max'].append(cmax)
        stats['area_mean'].append(am); stats['area_min'].append(amin); stats['area_max'].append(amax)

        if done % 100 == 0 or done == n_eval:
            rate = done / max(time.time() - t0, 1e-3)
            print(f"  [{done}/{n_eval}]  {rate:.1f} samples/s  "
                  f"L_gen={np.mean(stats['l_gen']):.4f}  "
                  f"col={np.mean(stats['col_mean']):.3f}  "
                  f"area={np.mean(stats['area_mean']):.3f}")

    # ── Aggregate ─────────────────────────────────────────────────────────
    r = {k: float(np.mean(v)) for k, v in stats.items()}
    r['mode_correct'] *= 100
    r['top5_correct'] *= 100
    r['median_gt_rank'] = int(np.median(stats['median_rank_input']))

    # ── Print ─────────────────────────────────────────────────────────────
    W = 78
    print("\n" + "=" * W)
    print(f"  OPEN-LOOP EVAL — INFRASTRUCTURE SANITY CHECK ({n_eval} samples)")
    print(f"  Cache: {args.cache}")
    print("=" * W)

    # Two metric tiers:
    #   (A) Hard pass/fail — must be exactly 0 or 100% on GT pass-through.
    #   (B) Practical ceilings — bounded by cache contents, threshold choices,
    #       and label noise. Report the value as the floor real-model evals
    #       should be measured against, like CLS-NR=93.08 for closed-loop.
    hard_rows = [
        ("L_gen",                    f"{r['l_gen']:.6f}", "0.000000", "exact",
         "pred == gt"),
        ("ADE (m)",                  f"{r['ade']:.6f}",   "0.000000", "exact",
         "pred == gt"),
        ("FDE (m)",                  f"{r['fde']:.6f}",   "0.000000", "exact",
         "pred == gt"),
        ("L_sel (CE w/ one-hot)",    f"{r['l_sel']:.6f}", "0.000000", "≈ 0",
         "logit_correct=+1e4, others=-1e4"),
        ("Mode acc top-1 (%)",       f"{r['mode_correct']:.2f}", "100.00", "exact",
         "argmax → mode_label"),
        ("Mode acc top-5 (%)",       f"{r['top5_correct']:.2f}", "100.00", "exact",
         ""),
        ("Median GT rank",           f"{r['median_gt_rank']}",   "1", "exact",
         ""),
        ("Consistent Ratio Lat (%)", f"{r['consistent_lat']:.2f}",
         f"100/N_LAT≈{100.0/cfg.N_LAT:.1f}", "exact",
         "all 60 modes = same GT traj → only GT's bin matches"),
        ("Consistent Ratio Lon (%)", f"{r['consistent_lon']:.2f}",
         f"100/N_LON≈{100.0/cfg.N_LON:.1f}", "exact",
         "same — only GT's speed bin matches"),
    ]
    ceiling_rows = [
        ("Open-loop Col Mean",       f"{r['col_mean']:.3f}",
         "GT vs other-agent radius (2 m). Real label noise — some GT "
         "frames have agents with-in 2 m even though the drive is fine."),
        ("Open-loop Area Mean",      f"{r['area_mean']:.3f}",
         "GT vs lane+route centerlines (3.5 m). Cache only stores "
         "N_LANES=20 closest at t=0; far-future timesteps lose coverage."),
    ]

    print(f"\n  Hard pass/fail (infrastructure must be perfect on GT):")
    print(f"  {'Metric':<28} {'Actual':>14} {'Expected':>14} {'Tol':>10}  Note")
    print(f"  {'─'*74}")
    for label, actual, expected, tol, note in hard_rows:
        print(f"  {label:<28} {actual:>14} {expected:>14} {tol:>10}  {note}")

    print(f"\n  Practical ceilings (record these as the baseline):")
    for label, actual, note in ceiling_rows:
        print(f"  {label:<28} {actual:>14}  {note}")
    print("=" * W)

    # ── Verdict ───────────────────────────────────────────────────────────
    fail = []
    if r['l_gen'] > 1e-4:    fail.append(f"L_gen={r['l_gen']:.6f} (expected 0)")
    if r['ade']   > 1e-4:    fail.append(f"ADE={r['ade']:.6f} (expected 0)")
    if r['fde']   > 1e-4:    fail.append(f"FDE={r['fde']:.6f} (expected 0)")
    if r['l_sel'] > 1e-2:    fail.append(f"L_sel={r['l_sel']:.4f} (expected ≈0)")
    if r['mode_correct'] < 99.99:    fail.append(f"top-1={r['mode_correct']:.2f}% (expected 100)")
    if r['top5_correct']  < 99.99:   fail.append(f"top-5={r['top5_correct']:.2f}% (expected 100)")
    if abs(r['consistent_lat'] - 100.0/cfg.N_LAT) > 0.01:
        fail.append(f"consistent_lat={r['consistent_lat']:.2f}% (expected {100.0/cfg.N_LAT:.2f})")
    if abs(r['consistent_lon'] - 100.0/cfg.N_LON) > 0.01:
        fail.append(f"consistent_lon={r['consistent_lon']:.2f}% (expected {100.0/cfg.N_LON:.2f})")

    if fail:
        print("\n[Sanity OL] FAIL — infrastructure issues detected:")
        for msg in fail:
            print(f"  - {msg}")
    else:
        print("\n[Sanity OL] PASS — open-loop infrastructure verified on hard metrics.")
        print(f"  Practical ceilings (real-model eval shortfalls below these are model issues):")
        print(f"    col_mean  ≥ {r['col_mean']:.3f}   (GT achieves this)")
        print(f"    area_mean ≥ {r['area_mean']:.3f}  (GT achieves this)")


def parse_args():
    p = argparse.ArgumentParser(description="Open-loop infra sanity check (GT pass-through)")
    p.add_argument('--split', default='val14', choices=list(SPLIT_TO_CACHE.keys()))
    p.add_argument('--cache', default=None,
                   help='Override cache path (default: auto from --split)')
    p.add_argument('--n_eval', type=int, default=0, help='0 = all samples')
    args = p.parse_args()
    if args.cache is None:
        args.cache = SPLIT_TO_CACHE[args.split]
    return args


if __name__ == '__main__':
    run(parse_args())
