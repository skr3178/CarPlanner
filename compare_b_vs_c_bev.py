"""
Side-by-side BEV comparison of Stage-B-seed vs Stage-C-best on the same val14
scenarios. Tests the hypothesis: did RL fine-tuning actually move the policy?

For each picked scenario, plots two panels showing the same map + GT trajectory
with B's best-mode prediction and C's best-mode prediction overlaid. Reports
mean L2 distance between B's and C's trajectories (small ⇒ RL barely changed
the policy mean).
"""
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as cfg
from data_loader import PreextractedDataset
from model import CarPlanner


def draw_panel(ax, sample, pred, all_trajs, gt, title, top_k=5, pad=5.0, min_window=12.0):
    map_lanes = sample['map_lanes'].cpu().numpy()
    map_lanes_mask = sample['map_lanes_mask'].cpu().numpy()
    agents_now = sample['agents_now'].cpu().numpy()
    agents_mask = sample['agents_history_mask'].cpu().numpy()

    # Lanes
    for i in range(len(map_lanes_mask)):
        if map_lanes_mask[i] < 0.5:
            continue
        lane = map_lanes[i]
        valid = np.abs(lane[:, 0]) + np.abs(lane[:, 1]) > 1e-6
        if valid.sum() < 2:
            continue
        c = lane[valid, 0:2]
        ax.plot(c[:, 0], c[:, 1], '-', color='#909090', linewidth=1.0, alpha=0.85, zorder=1)

    # Other agents
    for i in range(len(agents_mask)):
        if agents_mask[i] < 0.5:
            continue
        ax.plot(agents_now[i, 0], agents_now[i, 1], 's', color='#3b6ea8',
                markersize=5, alpha=0.7, zorder=2)

    # Top-K alt modes (faint)
    for j in range(min(top_k, all_trajs.shape[0])):
        t = all_trajs[j]
        ax.plot(t[:, 0], t[:, 1], '-', color='#7aa6e0', linewidth=0.7, alpha=0.4, zorder=3)

    # GT and prediction
    ax.plot(gt[:, 0], gt[:, 1], '-', color='#1f9e3c', linewidth=2.5, label='GT', zorder=5)
    ax.plot(pred[:, 0], pred[:, 1], '-', color='#d23a2c', linewidth=2.0, label='pred best', zorder=6)
    ax.plot(0, 0, '^', color='black', markersize=8, zorder=7)

    # Window
    pts = np.concatenate([gt[:, :2], pred[:, :2]], axis=0)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    rng = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), min_window) / 2 + pad
    ax.set_xlim(cx - rng, cx + rng)
    ax.set_ylim(cy - rng, cy + rng)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)


def run_model(model, sample, device):
    batch = {kk: v.unsqueeze(0).to(device) for kk, v in sample.items()}
    with torch.no_grad():
        mode_logits, all_trajs, best_traj, best_idx = model.forward_inference_fast(
            agents_now=batch['agents_now'],
            agents_mask=batch['agents_history_mask'],
            map_lanes=batch['map_lanes'],
            map_lanes_mask=batch['map_lanes_mask'],
            agents_history=batch['agents_history'],
            ego_history=batch.get('ego_history'),
            map_polygons=batch.get('map_polygons'),
            map_polygons_mask=batch.get('map_polygons_mask'),
            route_polylines=batch.get('route_polylines'),
            route_mask=batch.get('route_mask'),
        )
    return (mode_logits[0].cpu(),
            all_trajs[0].cpu().numpy(),
            best_traj[0].cpu().numpy(),
            int(best_idx.item()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_b', default='checkpoints/stage_b_20260424_140710/stage_b_epoch_020.pt')
    p.add_argument('--ckpt_c', default='checkpoints/stage_c_20260424_212311/stage_c_best.pt')
    p.add_argument('--cache', default='checkpoints/stage_cache_val14.pt')
    p.add_argument('--n_samples', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', default='compare_b_vs_c_bev.png')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Compare] Device: {device}')
    print(f'[Compare] Stage-B: {args.ckpt_b}')
    print(f'[Compare] Stage-C: {args.ckpt_c}')

    cfg.set_stage('b')
    dataset = PreextractedDataset(args.cache)
    print(f'[Compare] Loaded {len(dataset)} val14 samples')

    rng = np.random.default_rng(args.seed)
    indices = sorted(rng.choice(len(dataset), size=args.n_samples, replace=False).tolist())
    print(f'[Compare] Picked indices: {indices}')

    samples = [dataset[i] for i in indices]

    # Stage B
    model_b = CarPlanner().to(device)
    cb = torch.load(args.ckpt_b, map_location=device)
    model_b.load_state_dict(cb['model'], strict=False)
    model_b.eval()
    model_b._transition_loaded = True
    print(f"[Compare] Stage-B loaded (epoch {cb.get('epoch','?')})")

    out_b = [run_model(model_b, s, device) for s in samples]
    del model_b
    torch.cuda.empty_cache()

    # Stage C
    cfg.set_stage('c')
    model_c = CarPlanner().to(device)
    cc = torch.load(args.ckpt_c, map_location=device)
    model_c.load_state_dict(cc['model'], strict=False)
    model_c.eval()
    model_c._transition_loaded = True
    print(f"[Compare] Stage-C loaded (epoch {cc.get('epoch','?')})")

    out_c = [run_model(model_c, s, device) for s in samples]

    # Numeric: how different are B and C predictions?
    deltas, ade_b_list, ade_c_list = [], [], []
    for (logb, allb, predb, _), (logc, allc, predc, _), s in zip(out_b, out_c, samples):
        delta = np.linalg.norm(predb[:, :2] - predc[:, :2], axis=-1).mean()
        deltas.append(delta)
        gt = s['gt_trajectory'].cpu().numpy()
        ade_b_list.append(np.linalg.norm(predb[:, :2] - gt[:, :2], axis=-1).mean())
        ade_c_list.append(np.linalg.norm(predc[:, :2] - gt[:, :2], axis=-1).mean())

    mean_delta = float(np.mean(deltas))
    mean_ade_b = float(np.mean(ade_b_list))
    mean_ade_c = float(np.mean(ade_c_list))
    print()
    print(f'[Compare] Mean ‖predB - predC‖₂ over trajectory points: {mean_delta:.4f} m')
    print(f'[Compare] Mean ADE Stage-B: {mean_ade_b:.4f} m')
    print(f'[Compare] Mean ADE Stage-C: {mean_ade_c:.4f} m')
    if mean_delta < 0.05:
        print('[Compare] → predictions essentially identical: RL fine-tune did not move the policy mean')
    elif mean_delta < 0.5:
        print('[Compare] → predictions very close: RL barely moved the policy')
    else:
        print('[Compare] → predictions diverge meaningfully: RL did change the policy')

    # Plot
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4 * n))
    if n == 1:
        axes = axes[None, :]
    for r, ((logb, allb, predb, idxb),
            (logc, allc, predc, idxc), s, idx, d) in enumerate(
            zip(out_b, out_c, samples, indices, deltas)):
        gt = s['gt_trajectory'].cpu().numpy()
        ade_b = np.linalg.norm(predb[:, :2] - gt[:, :2], axis=-1).mean()
        ade_c = np.linalg.norm(predc[:, :2] - gt[:, :2], axis=-1).mean()
        gt_mode = int(s['mode_label'].item())
        title_b = f'#{idx}  Stage-B  mode {idxb} (GT={gt_mode})  ADE={ade_b:.2f}m'
        title_c = f'#{idx}  Stage-C  mode {idxc} (GT={gt_mode})  ADE={ade_c:.2f}m  ΔBC={d:.2f}m'
        draw_panel(axes[r, 0], s, predb, allb, gt, title_b)
        draw_panel(axes[r, 1], s, predc, allc, gt, title_c)
        if r == 0:
            axes[r, 0].legend(loc='upper left', fontsize=8)

    fig.suptitle(
        f'Stage-B-seed vs Stage-C  —  {n} val14 scenes\n'
        f'mean ‖predB-predC‖₂ = {mean_delta:.3f} m   '
        f'(ADE: B={mean_ade_b:.2f}m, C={mean_ade_c:.2f}m)',
        fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(args.output, dpi=140, bbox_inches='tight')
    print(f'[Compare] Saved: {args.output}')


if __name__ == '__main__':
    main()
