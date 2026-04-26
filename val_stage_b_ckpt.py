"""
Run a Stage B checkpoint against val14 and report L_IL components.
Mirrors the val loop in train_stage_b.py exactly.
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F

import config as cfg
from model import CarPlanner


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--val_cache', default='checkpoints/stage_cache_val14.pt')
    p.add_argument('--batch_size', type=int, default=512)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    cfg.set_stage('b')
    model = CarPlanner().to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    print(f"Loaded ckpt epoch={ckpt.get('epoch', '?')}  "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if ckpt.get('train_loss'):
        print(f"  ckpt train_loss: {ckpt['train_loss']}")
    model.eval()
    model._transition_loaded = True  # recompute β inside forward_train

    print(f"Loading val14 cache: {args.val_cache}")
    val = torch.load(args.val_cache, map_location=device)
    N = val['n_samples']
    print(f"  Val samples: {N}")

    def get(key, shape_default):
        return val.get(key, torch.zeros(shape_default, device=device))

    map_polygons = get('map_polygons',
        (N, cfg.N_POLYGONS, cfg.N_LANE_POINTS, cfg.D_POLYGON_POINT))
    map_polygons_mask = get('map_polygons_mask', (N, cfg.N_POLYGONS))
    route_polylines = get('route_polylines',
        (N, cfg.N_LAT, cfg.N_ROUTE_POINTS, cfg.D_POLYLINE_POINT))
    route_mask = get('route_mask', (N, cfg.N_LAT))

    losses = {'L_CE': 0., 'L_side': 0., 'L_gen': 0., 'L_total': 0.}
    losses_dim = torch.zeros(3, device=device)   # per-dim L1 (x, y, yaw)
    n_batches = 0
    BS = args.batch_size

    with torch.no_grad():
        for s in range(0, N, BS):
            e = min(s + BS, N)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                    enabled=device.type == 'cuda'):
                mode_logits, side_traj, pred_traj = model.forward_train(
                    val['agents_now'][s:e],
                    val['agents_mask'][s:e],
                    val['agents_seq'][s:e],  # ignored when _transition_loaded=True
                    val['gt_trajectory'][s:e],
                    val['mode_label'][s:e],
                    map_lanes=val['map_lanes'][s:e],
                    map_lanes_mask=val['map_lanes_mask'][s:e],
                    agents_history=val['agents_history'][s:e],
                    map_polygons=map_polygons[s:e],
                    map_polygons_mask=map_polygons_mask[s:e],
                    route_polylines=route_polylines[s:e],
                    route_mask=route_mask[s:e],
                )
                gt = val['gt_trajectory'][s:e]
                mode_label = val['mode_label'][s:e]
                L_CE = F.cross_entropy(mode_logits, mode_label)
                if cfg.SELECTOR_SIDE_TASK:
                    L_side = (side_traj - gt).abs().sum(dim=-1).mean()
                else:
                    L_side = torch.tensor(0.0, device=device)
                L_gen = (pred_traj - gt).abs().sum(dim=-1).mean()
                L_gen_dim = (pred_traj - gt).abs().float().mean(dim=(0, 1))   # (3,) — x, y, yaw
                L_total = L_CE + L_side + L_gen

            losses['L_CE']    += L_CE.item()
            losses['L_side']  += L_side.item()
            losses['L_gen']   += L_gen.item()
            losses['L_total'] += L_total.item()
            losses_dim        += L_gen_dim
            n_batches += 1

    avg = {k: v / n_batches for k, v in losses.items()}
    avg_dim = (losses_dim / n_batches).tolist()
    print(f"\nVal14 metrics on ckpt {os.path.basename(args.ckpt)}:")
    print(f"  L_total = {avg['L_total']:.4f}")
    print(f"  L_CE    = {avg['L_CE']:.4f}")
    print(f"  L_side  = {avg['L_side']:.4f}")
    print(f"  L_gen   = {avg['L_gen']:.4f}")
    print(f"    L_gen_x   = {avg_dim[0]:.4f}  [m]")
    print(f"    L_gen_y   = {avg_dim[1]:.4f}  [m]")
    print(f"    L_gen_yaw = {avg_dim[2]:.4f}  [rad]  ({avg_dim[2] * 180 / 3.14159:.2f}°)")
    print(f"    (sum={sum(avg_dim):.4f}, should match L_gen={avg['L_gen']:.4f})")
    print(f"  (n_batches={n_batches}, BS={BS})")


if __name__ == '__main__':
    main()
