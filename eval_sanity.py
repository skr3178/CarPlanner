"""
CarPlanner — Open-loop sanity check (BEV visualization + paper benchmark comparison).

For N samples, plots a bird's-eye-view panel showing:
  - map lane centerlines (gray)
  - other agents at t=0 (blue rectangles)
  - ground-truth future trajectory (green, thick)
  - best-mode predicted trajectory (red, thick, with heading arrows)
  - top-K alternative modes (faint blue)
  - ego heading arrow at t=0

Also reports aggregate metrics and compares them to paper benchmarks
(Table 4 open-loop L_gen, Table 1/2 closed-loop CLS-NR).
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

import config as cfg
from data_loader import PreextractedDataset
from model import CarPlanner


# Paper benchmarks — see paper/tables.md
PAPER_BENCHMARKS = {
    'IL': {
        'L_gen_open_loop': 174.3,     # Table 4, IL best config (Mode Drop + Side + Ego + BB)
        'L_sel_open_loop': 1.04,
        'CLS_NR_Test14_Random': 93.41,
        'CLS_NR_Test14_Hard': 47.50,
    },
    'IL+RL': {
        'L_gen_open_loop': 540.3,     # Table 4, RL best (Mode Drop + Side only)
        'L_sel_open_loop': 1.05,
        'CLS_NR_Test14_Random': 94.07,
        'CLS_NR_Test14_Hard': 95.14,
    },
}


def _draw_scene(ax, sample, pred, all_trajs, gt, mode_logits, chosen_mode, mode_label,
                ade, fde, sample_idx, top_k_plot=3, pad=5.0, min_window=12.0):
    """Draw a single BEV sanity panel."""
    # Map lanes — (N_LANES, N_PTS, 27) in ego frame; x,y are dims 0,1 of center polyline
    map_lanes = sample['map_lanes'].cpu().numpy()         # (L, P, 27)
    map_lanes_mask = sample['map_lanes_mask'].cpu().numpy()  # (L,)
    agents_now = sample['agents_now'].cpu().numpy()       # (N, 10)
    agents_mask = sample['agents_history_mask'].cpu().numpy()  # (N,)

    # Draw lanes (center = dims 0:2, left = 9:11, right = 18:20 per 27-dim polyline encoding)
    for i in range(len(map_lanes_mask)):
        if map_lanes_mask[i] < 0.5:
            continue
        lane = map_lanes[i]  # (P, 27)
        valid = np.abs(lane[:, 0]) + np.abs(lane[:, 1]) > 1e-6
        if valid.sum() < 2:
            continue
        center = lane[valid, 0:2]
        ax.plot(center[:, 0], center[:, 1], '-', color='#909090',
                linewidth=1.1, alpha=0.85, zorder=1)
        # Left / right boundaries if present
        if lane.shape[1] >= 21:
            left = lane[valid, 9:11]
            right = lane[valid, 18:20]
            if np.abs(left).sum() > 1e-3:
                ax.plot(left[:, 0], left[:, 1], '--', color='#bbbbbb',
                        linewidth=0.7, alpha=0.6, zorder=1)
            if np.abs(right).sum() > 1e-3:
                ax.plot(right[:, 0], right[:, 1], '--', color='#bbbbbb',
                        linewidth=0.7, alpha=0.6, zorder=1)

    # Other agents — small rectangles
    for n in range(len(agents_mask)):
        if agents_mask[n] < 0.5:
            continue
        ax_, ay = agents_now[n, 0], agents_now[n, 1]
        if abs(ax_) < 1e-6 and abs(ay) < 1e-6:
            continue  # ego at origin or padding
        yaw = agents_now[n, 2] if agents_now.shape[1] > 2 else 0.0
        ax.plot(ax_, ay, 's', color='steelblue', markersize=4, alpha=0.7, zorder=2)

    # All-mode spaghetti (top_k_plot most likely alternatives, faint)
    topk = torch.topk(mode_logits, min(top_k_plot, cfg.N_MODES)).indices.cpu().numpy()
    for m in topk:
        if m == chosen_mode:
            continue
        traj = all_trajs[m]
        ax.plot(traj[:, 0], traj[:, 1], '-', color='steelblue',
                alpha=0.25, linewidth=1.0, zorder=3)

    # Ground truth (green) and best prediction (red)
    ax.plot(gt[:, 0], gt[:, 1], '-', color='#2ca02c', linewidth=2.5, zorder=5,
            label='GT', marker='o', markersize=3)
    ax.plot(pred[:, 0], pred[:, 1], '-', color='#d62728', linewidth=2.0, zorder=6,
            label='Pred (best mode)', marker='s', markersize=3)

    # Ego at origin with heading arrow (assume ego oriented +x)
    ax.plot(0, 0, marker='*', color='black', markersize=12, zorder=7)
    ax.annotate('', xy=(1.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5), zorder=7)

    # Adaptive limits: center on union of gt+pred trajectories, min window of min_window metres
    all_xy = np.concatenate([gt[:, :2], pred[:, :2]], axis=0)
    cx, cy = all_xy.mean(axis=0)
    span = max(float(np.ptp(all_xy[:, 0])), float(np.ptp(all_xy[:, 1])), min_window)
    half = span / 2 + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25, zorder=0)

    mode_ok = '✓' if chosen_mode == mode_label else '✗'
    ax.set_title(
        f"#{sample_idx}  ADE={ade:.2f}m  FDE={fde:.2f}m\n"
        f"mode pred={chosen_mode} / gt={mode_label} {mode_ok}",
        fontsize=10
    )
    ax.set_xlabel('x (m, ego fwd)', fontsize=8)
    ax.set_ylabel('y (m, ego left)', fontsize=8)
    ax.tick_params(labelsize=7)


def _compute_sample_metrics(pred, gt):
    diff = pred[:, :2] - gt[:, :2]
    ade = float(np.mean(np.linalg.norm(diff, axis=1)))
    fde = float(np.linalg.norm(pred[-1, :2] - gt[-1, :2]))
    # L_gen (training convention): sum over state dims, mean over T
    l_gen_per_step = float(np.mean(np.sum(np.abs(pred - gt), axis=-1)))
    # L_gen (paper-like): sum over state dims AND time (per-sample total)
    l_gen_per_sample = float(np.sum(np.abs(pred - gt)))
    return ade, fde, l_gen_per_step, l_gen_per_sample


def _infer_stage(ckpt_path):
    name = os.path.basename(ckpt_path).lower()
    if 'stage_c' in name or 'rl' in name:
        return 'IL+RL'
    return 'IL'


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Eval] Device: {device}")

    dataset = PreextractedDataset(args.cache, device=device)
    print(f"[Eval] {len(dataset)} samples")

    model = CarPlanner().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    epoch = ckpt.get('epoch', '?')
    print(f"[Eval] Loaded {args.checkpoint} (epoch {epoch})")

    stage = args.stage or _infer_stage(args.checkpoint)
    print(f"[Eval] Stage: {stage}")

    n_plot = min(args.n_samples, len(dataset))
    plot_indices = np.linspace(0, len(dataset) - 1, n_plot, dtype=int)

    # Run aggregate eval over a larger pool for honest stats
    n_eval = min(args.n_eval, len(dataset))
    eval_indices = np.linspace(0, len(dataset) - 1, n_eval, dtype=int)
    eval_set = set(eval_indices.tolist())
    for i in plot_indices:
        eval_set.add(int(i))

    # Figure layout: N_rows x N_cols grid + header row for metrics
    ncols = 3
    nrows = (n_plot + ncols - 1) // ncols
    panel_size = 4.5
    fig_w = panel_size * ncols
    fig_h = panel_size * nrows + 2.5   # extra space for metric header

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[0.55] + [1] * nrows,
                          hspace=0.45, wspace=0.25)
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis('off')
    panel_axes = [fig.add_subplot(gs[r + 1, c]) for r in range(nrows) for c in range(ncols)]

    # Cache plot samples so we don't re-run inference for them during eval loop
    plot_cache = {}

    stats = {'ade': [], 'fde': [], 'l_gen_step': [], 'l_gen_sample': [], 'mode_correct': []}

    with torch.no_grad():
        for k, idx in enumerate(sorted(eval_set)):
            sample = dataset[idx]
            batch = {kk: v.unsqueeze(0).to(device) for kk, v in sample.items()}

            mode_logits, all_trajs, best_traj, best_idx = model.forward_inference(
                agents_now=batch['agents_now'],
                agents_mask=batch['agents_history_mask'],
                map_lanes=batch['map_lanes'],
                map_lanes_mask=batch['map_lanes_mask'],
                agents_history=batch['agents_history'],
            )

            gt = batch['gt_trajectory'][0].cpu().numpy()
            pred = best_traj[0].cpu().numpy()
            mode_label = batch['mode_label'].item()
            chosen_mode = best_idx.item()

            ade, fde, l_gen_step, l_gen_sample = _compute_sample_metrics(pred, gt)
            stats['ade'].append(ade)
            stats['fde'].append(fde)
            stats['l_gen_step'].append(l_gen_step)
            stats['l_gen_sample'].append(l_gen_sample)
            stats['mode_correct'].append(chosen_mode == mode_label)

            if idx in plot_indices:
                plot_cache[idx] = dict(
                    sample=sample, pred=pred, gt=gt,
                    all_trajs=all_trajs[0].cpu().numpy(),
                    mode_logits=mode_logits[0],
                    chosen_mode=chosen_mode, mode_label=mode_label,
                    ade=ade, fde=fde,
                )

            if (k + 1) % 50 == 0:
                print(f"  [{k+1}/{len(eval_set)}]")

    # Draw panels
    for plot_i, idx in enumerate(plot_indices):
        ax = panel_axes[plot_i]
        d = plot_cache[int(idx)]
        _draw_scene(
            ax, d['sample'], d['pred'], d['all_trajs'], d['gt'],
            d['mode_logits'], d['chosen_mode'], d['mode_label'],
            d['ade'], d['fde'], int(idx),
            top_k_plot=args.top_k_plot,
        )

    # Hide unused panels
    for j in range(len(plot_indices), len(panel_axes)):
        panel_axes[j].axis('off')

    # Single legend on first panel
    if plot_indices.size:
        panel_axes[0].legend(loc='upper left', fontsize=8, framealpha=0.9)

    # ── Header: metric comparison table ──
    ade_mean = float(np.mean(stats['ade']))
    fde_mean = float(np.mean(stats['fde']))
    lgen_step_mean = float(np.mean(stats['l_gen_step']))
    lgen_sample_mean = float(np.mean(stats['l_gen_sample']))
    mode_acc = float(np.mean(stats['mode_correct']))

    paper_il = PAPER_BENCHMARKS['IL']
    paper_rl = PAPER_BENCHMARKS['IL+RL']

    header_text = (
        f"Sanity Check — {os.path.basename(args.checkpoint)}  "
        f"(stage={stage}, epoch={epoch}, N={len(stats['ade'])} samples)\n"
    )
    header_ax.text(0.5, 0.92, header_text, transform=header_ax.transAxes,
                   ha='center', va='top', fontsize=13, weight='bold')

    table_rows = [
        ["Metric",                            "Ours",                     "Paper IL (Table 4)",             "Paper IL+RL (Table 4)"],
        ["ADE (m, best-of-K xy)",             f"{ade_mean:.3f}",          "— (not reported)",                "— (not reported)"],
        ["FDE (m, best-of-K xy)",             f"{fde_mean:.3f}",          "— (not reported)",                "— (not reported)"],
        ["L_gen per-step (train units)",      f"{lgen_step_mean:.3f}",    "— (not tabulated)",               "— (not tabulated)"],
        ["L_gen per-sample (paper units)",    f"{lgen_sample_mean:.2f}",  f"{paper_il['L_gen_open_loop']:.1f} (best cfg)",  f"{paper_rl['L_gen_open_loop']:.1f} (best cfg)"],
        ["Mode accuracy (top-1)",             f"{mode_acc*100:.2f}%",     f"L_sel={paper_il['L_sel_open_loop']:.2f}",        f"L_sel={paper_rl['L_sel_open_loop']:.2f}"],
        ["CLS-NR (closed-loop, separate eval)","see scripts/eval_closedloop_gpu", f"{paper_il['CLS_NR_Test14_Random']:.2f} / {paper_il['CLS_NR_Test14_Hard']:.2f} (Rnd/Hard)",  f"{paper_rl['CLS_NR_Test14_Random']:.2f} / {paper_rl['CLS_NR_Test14_Hard']:.2f} (Rnd/Hard)"],
    ]
    tbl = header_ax.table(cellText=table_rows, loc='lower center', cellLoc='center',
                          colWidths=[0.30, 0.20, 0.25, 0.25])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    # Style header row
    for c in range(len(table_rows[0])):
        tbl[(0, c)].set_facecolor('#d8d8d8')
        tbl[(0, c)].set_text_props(weight='bold')
    # Highlight "Ours" column
    for r in range(1, len(table_rows)):
        tbl[(r, 1)].set_facecolor('#fff4d6')

    plt.savefig(args.output, dpi=140, bbox_inches='tight')
    print(f"\n[Eval] Plot saved: {args.output}")

    # Terminal summary
    print(f"\n{'='*66}")
    print(f"  Aggregate stats over {len(stats['ade'])} samples ({stage})")
    print(f"  ADE:                 {ade_mean:.3f} m")
    print(f"  FDE:                 {fde_mean:.3f} m")
    print(f"  L_gen (per-step):    {lgen_step_mean:.3f}")
    print(f"  L_gen (per-sample):  {lgen_sample_mean:.2f}   vs paper: IL={paper_il['L_gen_open_loop']}, RL={paper_rl['L_gen_open_loop']}")
    print(f"  Mode accuracy:       {mode_acc*100:.2f}%")
    print(f"  [Closed-loop CLS-NR requires scripts/eval_closedloop_gpu.py]")
    print(f"{'='*66}")


def parse_args():
    p = argparse.ArgumentParser(description="CarPlanner sanity check — BEV viz + paper benchmarks")
    p.add_argument('--checkpoint', required=True, help='Path to checkpoint (.pt)')
    p.add_argument('--cache', default='checkpoints/stage_cache_mini.pt')
    p.add_argument('--n_samples', type=int, default=6, help='Scenes to plot')
    p.add_argument('--n_eval', type=int, default=200, help='Samples for aggregate metrics')
    p.add_argument('--top_k_plot', type=int, default=5, help='Alt modes to render faintly')
    p.add_argument('--output', default='eval_sanity.png')
    p.add_argument('--stage', default=None, choices=[None, 'IL', 'IL+RL'],
                   help='Override stage label (else inferred from filename)')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
