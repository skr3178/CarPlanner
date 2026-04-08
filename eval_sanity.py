"""
CarPlanner — Quick sanity check.
Loads a checkpoint, runs inference on cached samples, plots predicted vs GT.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as cfg
from data_loader import PreextractedDataset
from model import CarPlanner


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Eval] Device: {device}")

    # Load cache
    dataset = PreextractedDataset(args.cache, device=device)
    print(f"[Eval] {len(dataset)} samples")

    # Load model
    model = CarPlanner().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"[Eval] Loaded checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch','?')})")

    # Pick samples to visualize
    n_samples = min(args.n_samples, len(dataset))
    # Pick evenly spaced indices
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(2, (n_samples + 1) // 2, figsize=(5 * ((n_samples + 1) // 2), 10))
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    stats = {
        'ade': [], 'fde': [], 'pred_range': [],
        'gt_range': [], 'pred_dx': [], 'gt_dx': [],
        'best_mode': [], 'mode_correct': [],
    }

    with torch.no_grad():
        for plot_idx, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]
            # Add batch dim
            batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

            # Run inference
            mode_logits, all_trajs, best_traj, best_idx = model.forward_inference(
                agents_now=batch['agents_now'],
                agents_mask=batch['agents_history_mask'],
                map_lanes=batch['map_lanes'],
                map_lanes_mask=batch['map_lanes_mask'],
                agents_history=batch['agents_history'],
            )

            gt = batch['gt_trajectory'][0].cpu().numpy()       # (T, 3)
            pred = best_traj[0].cpu().numpy()                   # (T, 3)
            mode_label = batch['mode_label'].item()
            chosen_mode = best_idx.item()

            # Stats
            ade = np.mean(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1))
            fde = np.linalg.norm(pred[-1, :2] - gt[-1, :2])
            stats['ade'].append(ade)
            stats['fde'].append(fde)
            stats['pred_range'].append(pred[:, :2].max() - pred[:, :2].min())
            stats['gt_range'].append(gt[:, :2].max() - gt[:, :2].min())
            stats['pred_dx'].append(pred[-1, 0])
            stats['gt_dx'].append(gt[-1, 0])
            stats['best_mode'].append(chosen_mode)
            stats['mode_correct'].append(chosen_mode == mode_label)

            # Plot
            ax = axes[plot_idx]
            ax.plot(gt[:, 0], gt[:, 1], 'g-o', markersize=3, label='GT', linewidth=2)
            ax.plot(pred[:, 0], pred[:, 1], 'r-s', markersize=3, label='Pred', linewidth=2)

            # Plot start point
            ax.plot(0, 0, 'k*', markersize=10)

            # Also plot top-3 mode trajectories faintly
            top3 = torch.topk(mode_logits[0], min(3, cfg.N_MODES)).indices.cpu().numpy()
            all_trajs_np = all_trajs[0].cpu().numpy()  # (N_MODES, T, 3)
            for m in top3:
                ax.plot(all_trajs_np[m, :, 0], all_trajs_np[m, :, 1],
                        'b-', alpha=0.15, linewidth=1)

            ax.set_title(f"#{sample_idx}  ADE={ade:.2f}  mode={chosen_mode}/{mode_label}")
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            if plot_idx == 0:
                ax.legend(fontsize=8)

    plt.suptitle(f"Sanity Check — {os.path.basename(args.checkpoint)}", fontsize=14)
    plt.tight_layout()
    out_path = args.output
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n[Eval] Plot saved: {out_path}")

    # Print stats
    print(f"\n{'='*60}")
    print(f"  Stats over {n_samples} samples:")
    print(f"  ADE:           {np.mean(stats['ade']):.3f} ± {np.std(stats['ade']):.3f}")
    print(f"  FDE:           {np.mean(stats['fde']):.3f} ± {np.std(stats['fde']):.3f}")
    print(f"  Pred dx:       {np.mean(stats['pred_dx']):.2f} ± {np.std(stats['pred_dx']):.2f}")
    print(f"  GT dx:         {np.mean(stats['gt_dx']):.2f} ± {np.std(stats['gt_dx']):.2f}")
    print(f"  Pred spread:   {np.mean(stats['pred_range']):.2f}")
    print(f"  GT spread:     {np.mean(stats['gt_range']):.2f}")
    print(f"  Mode accuracy: {np.mean(stats['mode_correct'])*100:.1f}%")
    print(f"{'='*60}")

    # Quick diagnosis
    if np.mean(stats['pred_dx']) < 1.0:
        print("\n  WARNING: Predicted longitudinal displacement near zero — model may be collapsing!")
    if np.mean(stats['pred_range']) < 1.0:
        print("  WARNING: Predicted trajectories have almost no spatial spread — degenerate!")
    if np.mean(stats['ade']) > 20:
        print("  WARNING: ADE > 20m — predictions are far from ground truth.")


def parse_args():
    p = argparse.ArgumentParser(description="CarPlanner sanity check")
    p.add_argument('--checkpoint', required=True, help='Path to checkpoint (.pt)')
    p.add_argument('--cache', default='checkpoints/stage_cache_mini.pt')
    p.add_argument('--n_samples', type=int, default=10, help='Samples to visualize')
    p.add_argument('--output', default='eval_sanity.png')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
