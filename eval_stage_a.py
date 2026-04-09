"""
Stage A evaluation — tests trained TransitionModel checkpoint.

Three tests:
  1. L_tm vs stationary baseline  — model must beat predict-no-motion
  2. Predicted vs GT trajectory   — plots first 4 scenarios (saves PNG)
  3. Non-reactivity to ego        — moving ego should not change other agents' futures

Usage:
    python eval_stage_a.py --ckpt checkpoints/stage_a_best.pt --split mini --n_batches 50
"""

import os
import sys
import argparse

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as cfg
from data_loader import make_dataloader, make_cached_dataloader
from model import TransitionModel


# ── Loss (same as train_stage_a) ────────────────────────────────────────────

def compute_transition_loss(pred, gt, mask):
    diff = (pred[..., :8] - gt[..., :8]).abs().sum(dim=-1)  # (B, T, N) — motion dims only
    mask_t = mask.unsqueeze(1).expand_as(diff)
    return (diff * mask_t).sum() / (mask_t.sum() + 1e-6)


# ── Test 1: L_tm vs stationary baseline ─────────────────────────────────────

def test_loss_vs_baseline(model, loader, device, n_batches):
    print('\n' + '='*60)
    print('TEST 1 — L_tm vs stationary baseline')
    print('='*60)
    model.eval()
    total_model = 0.
    total_base  = 0.
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            agents_history = batch['agents_history'].to(device)   # (B, H, N, Da)
            agents_mask    = batch['agents_history_mask'].to(device)
            agents_seq     = batch['agents_seq'].to(device)        # (B, T, N, Da)
            map_lanes      = batch['map_lanes'].to(device)
            map_lanes_mask = batch['map_lanes_mask'].to(device)

            # Model prediction
            pred = model(agents_history, agents_mask, map_lanes, map_lanes_mask)

            # Stationary baseline: predict each agent stays at its t=0 position
            agents_now = agents_history[:, -1]                    # (B, N, Da) — last history frame
            baseline   = agents_now.unsqueeze(1).expand_as(agents_seq)

            L_model = compute_transition_loss(pred,     agents_seq, agents_mask).item()
            L_base  = compute_transition_loss(baseline, agents_seq, agents_mask).item()

            total_model += L_model
            total_base  += L_base
            count += 1

    avg_model = total_model / count
    avg_base  = total_base  / count
    improvement = (avg_base - avg_model) / avg_base * 100.

    print(f'  Model L_tm:       {avg_model:.4f}')
    print(f'  Stationary base:  {avg_base:.4f}')
    print(f'  Improvement:      {improvement:+.1f}%')
    if avg_model < avg_base:
        print('  PASS — model beats stationary baseline')
    else:
        print('  FAIL *** model does NOT beat stationary baseline')
    return avg_model, avg_base


# ── Test 2: Trajectory plot ──────────────────────────────────────────────────

def test_trajectory_plot(model, loader, device, save_path='checkpoints/stage_a_eval.png'):
    print('\n' + '='*60)
    print('TEST 2 — Predicted vs GT trajectory (first 4 scenarios)')
    print('='*60)
    model.eval()

    batch = next(iter(loader))
    agents_history = batch['agents_history'].to(device)
    agents_mask    = batch['agents_history_mask'].to(device)
    agents_seq     = batch['agents_seq'].to(device)
    map_lanes      = batch['map_lanes'].to(device)
    map_lanes_mask = batch['map_lanes_mask'].to(device)

    with torch.no_grad():
        pred = model(agents_history, agents_mask, map_lanes, map_lanes_mask)

    # pred, agents_seq: (B, T, N, Da) — plot first 4 batch items, first 5 valid agents
    B = min(4, pred.size(0))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for b in range(B):
        ax = axes[b]
        valid = agents_mask[b].bool()                 # (N,)
        n_valid = valid.sum().item()
        n_show  = min(5, n_valid)

        # GT: (T, N, 2) — x, y positions
        gt_xy   = agents_seq[b, :, :, :2].cpu()      # (T, N, 2)
        pred_xy = pred[b, :, :, :2].cpu()

        valid_idx = valid.nonzero(as_tuple=True)[0][:n_show]

        for k, idx in enumerate(valid_idx):
            color = f'C{k}'
            ax.plot(gt_xy[:, idx, 0],   gt_xy[:, idx, 1],
                    '-o', color=color, markersize=3, label=f'GT agent {idx}', alpha=0.8)
            ax.plot(pred_xy[:, idx, 0], pred_xy[:, idx, 1],
                    '--x', color=color, markersize=4, label=f'Pred agent {idx}', alpha=0.8)

        ax.set_title(f'Scenario {b+1}  ({n_valid} valid agents, showing {n_show})')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.legend(fontsize=6, ncol=2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Stage A: Predicted (--) vs GT (—) agent futures', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f'  Plot saved to: {save_path}')


# ── Test 3: Non-reactivity to ego ────────────────────────────────────────────

def test_non_reactivity(model, loader, device, n_batches=20):
    print('\n' + '='*60)
    print('TEST 3 — Non-reactivity to ego position')
    print('='*60)
    model.eval()

    total_diff_others = 0.
    total_diff_ego    = 0.
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            agents_history = batch['agents_history'].to(device)
            agents_mask    = batch['agents_history_mask'].to(device)
            map_lanes      = batch['map_lanes'].to(device)
            map_lanes_mask = batch['map_lanes_mask'].to(device)

            # Shift ego (agent 0) x,y by 50m across all history frames
            hist_moved = agents_history.clone()
            hist_moved[:, :, 0, :2] += 50.

            out_orig  = model(agents_history, agents_mask, map_lanes, map_lanes_mask)
            out_moved = model(hist_moved,     agents_mask, map_lanes, map_lanes_mask)

            # Non-ego agents: index 1 onwards
            diff_others = (out_orig[:, :, 1:] - out_moved[:, :, 1:]).abs().mean().item()
            diff_ego    = (out_orig[:, :, 0]  - out_moved[:, :, 0]).abs().mean().item()

            total_diff_others += diff_others
            total_diff_ego    += diff_ego
            count += 1

    avg_others = total_diff_others / count
    avg_ego    = total_diff_ego    / count
    ratio      = avg_others / (avg_ego + 1e-8)

    print(f'  Avg change in ego future:        {avg_ego:.4f}  (expected: nonzero)')
    print(f'  Avg change in other agent futures:{avg_others:.4f}  (expected: near 0)')
    print(f'  Non-reactivity ratio (others/ego):{ratio:.4f}  (lower is better, <0.1 is good)')

    if ratio < 0.1:
        print('  PASS — model is non-reactive to ego (ratio < 0.1)')
    elif ratio < 0.3:
        print('  PARTIAL — some ego leakage but acceptable (ratio < 0.3)')
    else:
        print('  FAIL *** strong ego leakage — β is reactive to ego')


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Stage A checkpoint evaluation')
    p.add_argument('--ckpt',      default='checkpoints/stage_a_best.pt')
    p.add_argument('--split',     default='mini')
    p.add_argument('--n_batches', type=int, default=50,
                   help='Number of batches to evaluate (default 50)')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--plot',      default='checkpoints/stage_a_eval.png')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'[Stage A Eval] Checkpoint: {args.ckpt}')
    print(f'[Stage A Eval] Device:     {device}')
    print(f'[Stage A Eval] Split:      {args.split}  Batches: {args.n_batches}')

    # Load model
    model = TransitionModel().to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    epoch = ckpt.get('epoch', '?')
    loss  = ckpt.get('loss',  '?')
    print(f'[Stage A Eval] Loaded epoch {epoch}, train L_tm={loss}')

    cache_path = os.path.join(cfg.CHECKPOINT_DIR, f"stage_cache_{args.split}.pt")
    if os.path.isfile(cache_path):
        print(f'[Stage A Eval] Using cache: {cache_path}')
        loader = make_cached_dataloader(
            cache_path, batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
    else:
        print(f'[Stage A Eval] No cache found, using slow DataLoader')
        loader = make_dataloader(
            args.split, batch_size=args.batch_size, shuffle=False, num_workers=2,
        )

    test_loss_vs_baseline(model, loader, device, args.n_batches)
    test_trajectory_plot(model, loader, device, args.plot)
    test_non_reactivity(model, loader, device, args.n_batches)

    print('\n[Stage A Eval] Done.')


if __name__ == '__main__':
    main()
