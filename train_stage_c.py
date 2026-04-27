"""
CarPlanner — RL fine-tuning (Stage C).

Loss (Algorithm 1, lines 23-27, 34):
  L_generator = λ_policy*L_policy + λ_value*L_value - λ_entropy*L_entropy
  L = L_selector + L_generator

  L_policy   = PPO clip loss (Eq 8), λ_policy=100
  L_value    = MSE value loss (Eq 9), λ_value=3
  L_entropy  = entropy bonus (Eq 10), λ_entropy=0.001
  L_selector = CrossEntropy + SideTask (same as Stage B, line 21)

Optimizer: AdamW, lr=1e-4.  PPO with I=8 update interval (Table 5).

Usage:
    python train_stage_c.py --split mini \
        --stage_a_ckpt checkpoints/stage_a_best.pt \
        --stage_b_ckpt checkpoints/best.pt \
        --epochs 50
"""

import os
import sys
import argparse
import copy
import random
import time

# Flush stdout immediately so nohup log updates in real time
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import torch
import torch.nn.functional as F

import config as cfg
from data_loader import make_dataloader, make_cached_dataloader
from model import CarPlanner
from rewards import compute_rewards, compute_gae, normalize_advantages


# ── Loss functions ─────────────────────────────────────────────────────────────

def compute_ppo_loss(log_probs_new, log_probs_old, advantages, epsilon=cfg.PPO_CLIP):
    """PPO clipped policy loss (Eq 8)."""
    ratio = torch.exp(log_probs_new - log_probs_old)       # (B, T)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    return -torch.min(surr1, surr2).mean()


def compute_value_loss(values_new, returns):
    """Value function MSE loss (Eq 9)."""
    return F.mse_loss(values_new, returns)


def compute_entropy_loss(entropies):
    """Negative mean entropy (Eq 10) — subtracted in total loss → maximises entropy."""
    return -entropies.mean()


def compute_selector_loss(mode_logits, side_traj, gt_traj, mode_label):
    """Mode selector loss (same as Stage B)."""
    L_CE = F.cross_entropy(mode_logits, mode_label)
    if cfg.SELECTOR_SIDE_TASK:
        L_side = (side_traj - gt_traj).abs().sum(dim=-1).mean()
    else:
        L_side = torch.tensor(0.0, device=mode_logits.device)
    return L_CE + L_side, {'L_CE': L_CE.item(), 'L_side': L_side.item()}


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Stage C] RL Fine-tuning")
    print(f"[Stage C] Device: {device}")
    print(f"[Stage C] Split: {args.split}, Epochs: {args.epochs}, "
          f"Batch: {args.batch_size}")

    # ── Data — cache-first, slow fallback ──────────────────────────────────
    cache_path = os.path.join(
        cfg.CHECKPOINT_DIR, f"stage_cache_{args.split}.pt"
    )
    use_cache = os.path.isfile(cache_path)

    pin_gpu = args.pin_gpu and device.type == 'cuda'
    if use_cache:
        print(f"[Stage C] Loading pre-extracted cache: {cache_path}")
        loader = make_cached_dataloader(
            cache_path,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        all_batches = list(loader)
        if pin_gpu:
            print(f"[Stage C] Pinning all batches to GPU...")
            all_batches = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in b.items()}
                for b in all_batches
            ]
            print(f"[Stage C] GPU cache ready. "
                  f"~{torch.cuda.memory_allocated()/1e9:.1f}GB GPU used")
        print(f"[Stage C] Cache loaded: {len(loader.dataset)} samples, "
              f"{len(all_batches)} batches")
    else:
        print(f"[Stage C] WARNING: No cache found. Falling back to slow DataLoader.")
        loader = make_dataloader(
            args.split,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            max_per_file=args.max_per_file,
        )
        all_batches = None
        print(f"[Stage C] Dataset size: {len(loader.dataset)}, "
              f"Batches/epoch: {len(loader)}")

    # ── Model ───────────────────────────────────────────────────────────────
    cfg.set_stage('c')
    model = CarPlanner().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Stage C] Parameters: {n_params:,}")

    # Load frozen transition model from Stage A
    if args.stage_a_ckpt:
        model.load_transition_model(args.stage_a_ckpt, freeze=True)
        print(f"[Stage C] Loaded frozen transition model: {args.stage_a_ckpt}")
    else:
        raise ValueError("--stage_a_ckpt is required for RL training")

    # Load Stage B checkpoint → warm-start policy + selector
    if args.stage_b_ckpt:
        ckpt_b = torch.load(args.stage_b_ckpt, map_location=device)
        pretrained = ckpt_b['model']
        # Rename action_head → action_mean_head for Stage C compatibility
        model_dict = model.state_dict()
        # Only load keys that exist in both checkpoint and current model
        loaded = {k: v for k, v in pretrained.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(loaded)
        model.load_state_dict(model_dict)
        n_loaded = len(loaded)
        n_total = len(model_dict)
        print(f"[Stage C] Loaded {n_loaded}/{n_total} params from: {args.stage_b_ckpt}")
        print(f"[Stage C]   New params (value_head, action_log_std) randomly initialised")
    else:
        raise ValueError("--stage_b_ckpt is required for RL training")

    # ── Old policy for PPO ──────────────────────────────────────────────────
    policy_old = copy.deepcopy(model.policy)
    for p in policy_old.parameters():
        p.requires_grad = False
    policy_old.eval()

    # ── Optimiser & scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=cfg.LR_PATIENCE, factor=cfg.LR_FACTOR,
    )

    # Resume from Stage C checkpoint if provided
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt_c = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt_c['model'])
        optimizer.load_state_dict(ckpt_c['optimizer'])
        policy_old.load_state_dict(ckpt_c['policy_old'])
        start_epoch = ckpt_c['epoch'] + 1
        print(f"[Stage C] Resumed from epoch {start_epoch}")

    # ── Val14 cache ──────────────────────────────────────────────────────────
    val_cache_path = os.path.join(cfg.CHECKPOINT_DIR, "stage_cache_val14.pt")
    if os.path.isfile(val_cache_path):
        val_loader = make_cached_dataloader(
            val_cache_path, batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        val_batches = list(val_loader)
        if pin_gpu:
            val_batches = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in b.items()}
                for b in val_batches
            ]
        print(f"[Stage C] Val14 loaded: {len(val_loader.dataset)} samples, "
              f"{len(val_batches)} batches")
    else:
        val_batches = None
        print(f"[Stage C] No val14 cache found at {val_cache_path}, skipping validation")

    # ── Checkpoint dir ─────────────────────────────────────────────────────
    from datetime import datetime
    ckpt_dir = os.path.join(
        cfg.CHECKPOINT_DIR,
        f"stage_c_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float('inf')
    update_interval = 8  # Paper Table 5: I=8
    step_counter = 0

    # ── Epoch loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {
            'L_policy': 0., 'L_value': 0., 'L_entropy': 0.,
            'L_selector': 0., 'L_total': 0.,
        }
        t0 = time.time()

        for batch_idx, batch in enumerate(random.sample(all_batches, len(all_batches)) if use_cache else loader):
            agents_now      = batch['agents_now'] if pin_gpu else batch['agents_now'].to(device)
            agents_history  = batch['agents_history'] if pin_gpu else batch['agents_history'].to(device)
            agents_mask     = batch['agents_history_mask'] if pin_gpu else batch['agents_history_mask'].to(device)
            gt_traj         = batch['gt_trajectory'] if pin_gpu else batch['gt_trajectory'].to(device)
            mode_label      = batch['mode_label'] if pin_gpu else batch['mode_label'].to(device)
            map_lanes       = batch['map_lanes'] if pin_gpu else batch['map_lanes'].to(device)
            map_lanes_mask  = batch['map_lanes_mask'] if pin_gpu else batch['map_lanes_mask'].to(device)
            map_polygons    = (batch['map_polygons'] if pin_gpu else batch['map_polygons'].to(device)) if 'map_polygons' in batch else None
            map_polygons_mask = (batch['map_polygons_mask'] if pin_gpu else batch['map_polygons_mask'].to(device)) if 'map_polygons_mask' in batch else None
            route_polylines = (batch['route_polylines'] if pin_gpu else batch['route_polylines'].to(device)) if 'route_polylines' in batch else None
            route_mask      = (batch['route_mask'] if pin_gpu else batch['route_mask'].to(device)) if 'route_mask' in batch else None
            ego_history     = (batch['ego_history'] if pin_gpu else batch['ego_history'].to(device)) if 'ego_history' in batch else None

            # Step 1: Simulate agents with frozen transition model
            with torch.no_grad():
                agent_futures = model.transition_model(
                    agents_history, agents_mask,
                    map_lanes, map_lanes_mask,
                    map_polygons=map_polygons, map_polygons_mask=map_polygons_mask,
                )                                            # (B, T, N, Da)

            # PPO ratio is meaningful only if old and new policies see the SAME
            # state features, so encode ego_history → ego_token once with a
            # frozen dropout decision and share between policy_old and forward_rl_train.
            ego_token = None
            if ego_history is not None:
                ego_for_enc = ego_history
                if cfg.EGO_HISTORY_DROPOUT and model.training:
                    keep = (torch.rand(ego_history.size(0), 1, 1, device=ego_history.device)
                            >= cfg.MODE_DROPOUT_P).float()
                    ego_for_enc = ego_history * keep
                ego_token = model.ego_encoder(ego_for_enc)

            # Step 2: Rollout with π_old (no grad)
            with torch.no_grad():
                traj_old, log_probs_old, values_old, _ = policy_old.forward_rl(
                    agents_now=agents_now,
                    agents_seq=agent_futures,
                    agents_mask=agents_mask,
                    mode_c=mode_label,
                    map_lanes=map_lanes,
                    map_lanes_mask=map_lanes_mask,
                    map_polygons=map_polygons,
                    map_polygons_mask=map_polygons_mask,
                    route_polylines=route_polylines,
                    route_mask=route_mask,
                    ego_token=ego_token,
                )

            # Step 3: Compute rewards + GAE
            with torch.no_grad():
                rewards = compute_rewards(
                    ego_traj=traj_old, gt_traj=gt_traj,
                    agent_futures=agent_futures, agents_mask=agents_mask,
                    map_lanes=map_lanes, map_lanes_mask=map_lanes_mask,
                )
                advantages, returns = compute_gae(rewards, values_old)
                advantages = normalize_advantages(advantages)

            # Step 4: Recompute with current π (with grad). Reuse the same ego_token
            # so old and new differ only in policy params, not in state features.
            mode_logits, side_traj, traj_new, log_probs_new, values_new, entropy = \
                model.forward_rl_train(
                    agents_now=agents_now,
                    agents_mask=agents_mask,
                    agents_seq=agent_futures,
                    gt_traj=gt_traj,
                    mode_label=mode_label,
                    map_lanes=map_lanes,
                    map_lanes_mask=map_lanes_mask,
                    stored_actions=traj_old,
                    agents_history=agents_history,
                    ego_token=ego_token,
                    map_polygons=map_polygons,
                    map_polygons_mask=map_polygons_mask,
                    route_polylines=route_polylines,
                    route_mask=route_mask,
                )

            # Step 5: Losses (Algorithm 1 lines 21, 27, 34)
            L_policy   = compute_ppo_loss(log_probs_new, log_probs_old, advantages)
            L_value    = compute_value_loss(values_new, returns)
            L_entropy  = compute_entropy_loss(entropy)
            L_selector, _ = compute_selector_loss(
                mode_logits, side_traj, gt_traj, mode_label
            )

            L_generator = (cfg.LAMBDA_POLICY * L_policy
                           + cfg.LAMBDA_VALUE * L_value
                           + cfg.LAMBDA_ENTROPY * L_entropy)
            L_total = L_selector + L_generator

            # Backward
            optimizer.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Accumulate
            for k in epoch_losses:
                epoch_losses[k] += locals()[k].item() if k != 'L_total' else L_total.item()

            # Logging
            n_batches = len(all_batches) if use_cache else len(loader)
            if (batch_idx + 1) % max(1, n_batches // 5) == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}  "
                      f"[{batch_idx+1}/{n_batches}]  "
                      f"L_total={L_total.item():.4f}  "
                      f"L_policy={L_policy.item():.4f}  "
                      f"L_value={L_value.item():.4f}")

            # Step 6: Update π_old every I=8 steps (Alg 1 line 38-39, Table 5)
            step_counter += 1
            if step_counter % update_interval == 0:
                policy_old = copy.deepcopy(model.policy)
                for p in policy_old.parameters():
                    p.requires_grad = False
                policy_old.eval()

        # Epoch summary
        n = len(all_batches) if use_cache else len(loader)
        avg = {k: v / n for k, v in epoch_losses.items()}
        elapsed = time.time() - t0

        # ── Validation ─────────────────────────────────────────────────────
        val_avg = {}
        if val_batches is not None:
            model.eval()
            val_losses = {k: 0. for k in epoch_losses}
            with torch.no_grad():
                for vb in val_batches:
                    v_agents_now     = vb['agents_now'] if pin_gpu else vb['agents_now'].to(device)
                    v_agents_history = vb['agents_history'] if pin_gpu else vb['agents_history'].to(device)
                    v_agents_mask    = vb['agents_history_mask'] if pin_gpu else vb['agents_history_mask'].to(device)
                    v_gt_traj        = vb['gt_trajectory'] if pin_gpu else vb['gt_trajectory'].to(device)
                    v_mode_label     = vb['mode_label'] if pin_gpu else vb['mode_label'].to(device)
                    v_map_lanes      = vb['map_lanes'] if pin_gpu else vb['map_lanes'].to(device)
                    v_map_lanes_mask = vb['map_lanes_mask'] if pin_gpu else vb['map_lanes_mask'].to(device)
                    v_map_polygons   = (vb['map_polygons'] if pin_gpu else vb['map_polygons'].to(device)) if 'map_polygons' in vb else None
                    v_map_poly_mask  = (vb['map_polygons_mask'] if pin_gpu else vb['map_polygons_mask'].to(device)) if 'map_polygons_mask' in vb else None
                    v_route_poly     = (vb['route_polylines'] if pin_gpu else vb['route_polylines'].to(device)) if 'route_polylines' in vb else None
                    v_route_mask     = (vb['route_mask'] if pin_gpu else vb['route_mask'].to(device)) if 'route_mask' in vb else None
                    v_ego_history    = (vb['ego_history'] if pin_gpu else vb['ego_history'].to(device)) if 'ego_history' in vb else None

                    v_agent_futures = model.transition_model(
                        v_agents_history, v_agents_mask,
                        v_map_lanes, v_map_lanes_mask,
                        map_polygons=v_map_polygons, map_polygons_mask=v_map_poly_mask,
                    )

                    # Encode ego_history once and share between old/new paths
                    # (same rationale as training: keep PPO ratio meaningful).
                    v_ego_token = (model.ego_encoder(v_ego_history)
                                   if v_ego_history is not None else None)

                    # Old log_probs must come from policy_old, not model.policy —
                    # otherwise the PPO ratio collapses to 1 in val.
                    v_traj, v_lp, v_vals, _ = policy_old.forward_rl(
                        agents_now=v_agents_now, agents_seq=v_agent_futures,
                        agents_mask=v_agents_mask, mode_c=v_mode_label,
                        map_lanes=v_map_lanes, map_lanes_mask=v_map_lanes_mask,
                        map_polygons=v_map_polygons, map_polygons_mask=v_map_poly_mask,
                        route_polylines=v_route_poly, route_mask=v_route_mask,
                        ego_token=v_ego_token,
                    )

                    v_rewards = compute_rewards(
                        ego_traj=v_traj, gt_traj=v_gt_traj,
                        agent_futures=v_agent_futures, agents_mask=v_agents_mask,
                        map_lanes=v_map_lanes, map_lanes_mask=v_map_lanes_mask,
                    )
                    v_adv, v_ret = compute_gae(v_rewards, v_vals)
                    v_adv = normalize_advantages(v_adv)

                    v_ml, v_st, v_tn, v_lpn, v_vn, v_ent = model.forward_rl_train(
                        agents_now=v_agents_now, agents_mask=v_agents_mask,
                        agents_seq=v_agent_futures, gt_traj=v_gt_traj,
                        mode_label=v_mode_label,
                        map_lanes=v_map_lanes, map_lanes_mask=v_map_lanes_mask,
                        stored_actions=v_traj, agents_history=v_agents_history,
                        ego_token=v_ego_token,
                        map_polygons=v_map_polygons, map_polygons_mask=v_map_poly_mask,
                        route_polylines=v_route_poly, route_mask=v_route_mask,
                    )

                    vL_policy   = compute_ppo_loss(v_lpn, v_lp, v_adv)
                    vL_value    = compute_value_loss(v_vn, v_ret)
                    vL_entropy  = compute_entropy_loss(v_ent)
                    vL_selector, _ = compute_selector_loss(v_ml, v_st, v_gt_traj, v_mode_label)
                    vL_generator = (cfg.LAMBDA_POLICY * vL_policy
                                    + cfg.LAMBDA_VALUE * vL_value
                                    + cfg.LAMBDA_ENTROPY * vL_entropy)
                    vL_total = vL_selector + vL_generator

                    val_losses['L_policy']   += vL_policy.item()
                    val_losses['L_value']    += vL_value.item()
                    val_losses['L_entropy']  += vL_entropy.item()
                    val_losses['L_selector'] += vL_selector.item()
                    val_losses['L_total']    += vL_total.item()

            val_avg = {k: v / len(val_batches) for k, v in val_losses.items()}
            model.train()

        val_str = f"  val={val_avg['L_total']:.4f}" if val_avg else ""
        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"train={avg['L_total']:.4f}  "
              f"L_policy={avg['L_policy']:.4f}  "
              f"L_value={avg['L_value']:.4f}  "
              f"L_entropy={avg['L_entropy']:.4f}  "
              f"L_selector={avg['L_selector']:.4f}"
              f"{val_str}  ({elapsed:.1f}s)")

        track_loss = val_avg.get('L_total', avg['L_total'])
        scheduler.step(track_loss)

        # Save checkpoint
        is_best = track_loss < best_val_loss
        best_val_loss = min(track_loss, best_val_loss)

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'policy_old': policy_old.state_dict(),
            'train_loss': avg,
            'val_loss': val_avg,
        }
        ckpt_path = os.path.join(ckpt_dir, f"stage_c_epoch_{epoch+1:03d}.pt")
        torch.save(ckpt, ckpt_path)
        if is_best:
            best_path = os.path.join(ckpt_dir, "stage_c_best.pt")
            torch.save(ckpt, best_path)
            torch.save(ckpt, os.path.join(cfg.CHECKPOINT_DIR, "stage_c_best.pt"))
            print(f"  * New best saved: {best_path}")

    print(f"\n[Stage C] Done. Best val L_total={best_val_loss:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CarPlanner RL fine-tuning (Stage C)")
    p.add_argument('--split', default='mini',
                   choices=['mini', 'train_boston', 'train_pittsburgh', 'train_singapore',
                            'train_all', 'train_all_balanced', 'train_4city_balanced',
                            'train_vegas_balanced'])
    p.add_argument('--epochs', type=int, default=cfg.EPOCHS)
    p.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE)
    p.add_argument('--num_workers', type=int, default=cfg.NUM_WORKERS)
    p.add_argument('--max_per_file', type=int, default=None)
    p.add_argument('--stage_a_ckpt', required=True,
                   help='Path to Stage A checkpoint (transition model)')
    p.add_argument('--stage_b_ckpt', required=True,
                   help='Path to Stage B checkpoint (policy + selector)')
    p.add_argument('--resume', default=None,
                   help='Path to Stage C checkpoint to resume from')
    p.add_argument('--pin_gpu', action='store_true',
                   help='Pre-load entire cache to GPU for faster training')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
