"""
CarPlanner baseline — open-loop evaluation.

Metric: best-of-K L2 displacement error (Section 4.1 / requirements.md).
  For each sample:
    1. Score all N_MODES modes via mode selector
    2. Generate trajectories for top-K modes only (or all if --top_k not set)
    3. Compute L2 = mean_t ||pred_xy_t - gt_xy_t||_2  for each mode
    4. Take best-of-K = min over generated modes
  Report mean best-of-K L2 over the dataset.
"""

import os
import sys
import argparse

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import torch
import torch.nn.functional as F

import config as cfg
from data_loader import make_dataloader, make_cached_dataloader
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

    cache_path = os.path.join(cfg.CHECKPOINT_DIR, f"stage_cache_{args.split}.pt")
    if args.cache and os.path.isfile(cache_path):
        print(f"[Eval] Using cached dataset: {cache_path}")
        loader = make_cached_dataloader(
            cache_path,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        if args.cache:
            print(f"[Eval] Cache not found at {cache_path}, falling back to raw data loader.")
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
    n_modes = cfg.N_MODES

    n_batches = len(loader)
    print(f"[Eval] Modes: {n_modes} (batched single policy call)")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{n_batches}]")
            agents_now  = batch['agents_now'].to(device)
            agents_mask = batch['agents_history_mask'].to(device)
            gt_traj     = batch['gt_trajectory'].to(device)
            mode_label  = batch['mode_label'].to(device)
            map_lanes   = batch['map_lanes'].to(device)
            map_lanes_mask = batch['map_lanes_mask'].to(device)
            B = agents_now.size(0)

            # --- Manual inference with top-k ---
            # Step 1-2: encode + mode scores
            s0_per_agent, s0_global = model.s0_encoder(agents_now, agents_mask)
            mode_logits, _ = model.mode_selector(
                s0_global, s0_per_agent=s0_per_agent,
                map_lanes=map_lanes, map_lanes_mask=map_lanes_mask,
            )

            # Step 3: transition model → agent futures
            hist = batch.get('agents_history', agents_now.unsqueeze(1)).to(device)
            agent_futures = model.transition_model(
                hist, agents_mask, map_lanes, map_lanes_mask
            )  # (B, T, N, Da)

            # Step 4: generate trajectories — batched across all modes
            # Repeat inputs along batch dim: (B, ...) → (B*M, ...) where M = n_modes
            M = n_modes
            agents_now_exp   = agents_now.repeat_interleave(M, dim=0)    # (B*M, N, Da)
            agents_mask_exp  = agents_mask.repeat_interleave(M, dim=0)   # (B*M, N)
            agent_futures_exp = agent_futures.repeat_interleave(M, dim=0) # (B*M, T, N, Da)
            map_lanes_exp    = map_lanes.repeat_interleave(M, dim=0)     # (B*M, L, P, F)
            map_lanes_mask_exp = map_lanes_mask.repeat_interleave(M, dim=0) # (B*M, L)

            # Mode indices: [0,1,...,M-1] repeated for each sample
            mode_c = torch.arange(M, device=device).unsqueeze(0).expand(B, M).reshape(-1)  # (B*M,)

            # Single batched policy call — fills the GPU
            all_trajs_flat = model.policy(
                agents_now=agents_now_exp,
                agents_seq=agent_futures_exp,
                agents_mask=agents_mask_exp,
                mode_c=mode_c,
                gt_ego=None,
                map_lanes=map_lanes_exp,
                map_lanes_mask=map_lanes_mask_exp,
            )  # (B*M, T, 3)

            # Reshape back: (B*M, T, 3) → (B, M, T, 3)
            all_trajs = all_trajs_flat.view(B, M, -1, 3)

            # Free the expanded tensors immediately
            del agents_now_exp, agents_mask_exp, agent_futures_exp
            del map_lanes_exp, map_lanes_mask_exp, all_trajs_flat

            # L2 displacement per generated mode
            l2_per_mode = l2_displacement(all_trajs, gt_traj)   # (B, n_modes)

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
    print(f"  Best-of-{n_modes} L2 displacement:  {mean_best_l2:.4f} m")
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
    p.add_argument('--cache', action='store_true',
                   help='Use pre-extracted .pt cache if available (fast)')
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
