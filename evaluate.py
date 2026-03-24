"""
CarPlanner baseline — open-loop evaluation.

Metric: best-of-K L2 displacement error (Section 4.1 / requirements.md).
  For each sample:
    1. Generate trajectories for all N_MODES modes
    2. Compute L2 = mean_t ||pred_xy_t - gt_xy_t||_2  for each mode
    3. Take best-of-K = min over modes
  Report mean best-of-K L2 over the dataset.
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F

import config as cfg
from data_loader import make_dataloader
from model import CarPlanner


def l2_displacement(pred_traj, gt_traj):
    """
    Compute per-sample, per-mode L2 displacement error.
    pred_traj: (B, N_MODES, T, 3)  — last dim is (x, y, yaw)
    gt_traj:   (B, T, 3)
    Returns:   (B, N_MODES) — mean L2 over T steps for xy only
    """
    gt_exp = gt_traj[:, None, :, :2].expand_as(pred_traj[..., :2])  # (B, M, T, 2)
    diff = pred_traj[..., :2] - gt_exp                               # (B, M, T, 2)
    dist = diff.norm(dim=-1)                                         # (B, M, T)
    return dist.mean(dim=-1)                                         # (B, M)


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Eval] Device: {device}, Split: {args.split}")

    loader = make_dataloader(
        args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        max_per_file=args.max_per_file,
    )
    print(f"[Eval] Dataset size: {len(loader.dataset)}")

    # Load model
    model = CarPlanner().to(device)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"[Eval] Loaded checkpoint: {args.checkpoint}")
    else:
        print("[Eval] No checkpoint provided — evaluating random init model.")

    model.eval()

    all_best_l2 = []
    all_mode_acc = []

    with torch.no_grad():
        for batch in loader:
            agents_now  = batch['agents_now'].to(device)
            agents_mask = batch['agents_history_mask'].to(device)
            gt_traj     = batch['gt_trajectory'].to(device)
            mode_label  = batch['mode_label'].to(device)
            map_lanes   = batch['map_lanes'].to(device)
            map_lanes_mask = batch['map_lanes_mask'].to(device)

            # Generate all N_MODES trajectories + rule-selected best
            mode_logits, all_trajs, best_traj, best_idx = model.forward_inference(
                agents_now, agents_mask,
                map_lanes=map_lanes, map_lanes_mask=map_lanes_mask
            )
            # all_trajs: (B, N_MODES, T, 3)

            # L2 displacement per mode
            l2_per_mode = l2_displacement(all_trajs, gt_traj)   # (B, N_MODES)

            # Best-of-K
            best_l2 = l2_per_mode.min(dim=1).values             # (B,)
            all_best_l2.append(best_l2)

            # Mode accuracy (is argmax logit == mode_label?)
            pred_mode = mode_logits.argmax(dim=1)                # (B,)
            acc = (pred_mode == mode_label).float()
            all_mode_acc.append(acc)

    all_best_l2 = torch.cat(all_best_l2)
    all_mode_acc = torch.cat(all_mode_acc)

    mean_best_l2 = all_best_l2.mean().item()
    mode_accuracy = all_mode_acc.mean().item()

    print(f"\n=== Evaluation Results ({args.split}) ===")
    print(f"  Best-of-{cfg.N_MODES} L2 displacement:  {mean_best_l2:.4f} m")
    print(f"  Mode prediction accuracy:             {mode_accuracy*100:.2f}%")
    print(f"  Samples evaluated:                    {len(all_best_l2)}")

    # Per-mode analysis
    return mean_best_l2, mode_accuracy


def parse_args():
    p = argparse.ArgumentParser(description="CarPlanner open-loop evaluation")
    p.add_argument('--split', default='mini',
                   choices=['mini', 'train_boston'])
    p.add_argument('--checkpoint', default=None,
                   help='Path to model checkpoint (e.g. checkpoints/best.pt)')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=cfg.NUM_WORKERS)
    p.add_argument('--max_per_file', type=int, default=None)
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
