"""Diagnose whether the rule-augmented selector is overriding correct mode choices."""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

import config as cfg
from data_loader import PreextractedDataset
from model import CarPlanner


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PreextractedDataset(args.cache, device=device)
    n = min(args.n_eval, len(dataset)) if args.n_eval > 0 else len(dataset)
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int).tolist()
    loader = DataLoader(torch.utils.data.Subset(dataset, indices),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CarPlanner().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"[Diag] {args.checkpoint} (epoch {ckpt.get('epoch', '?')})  on  {args.cache}  ({n} samples)")

    n_logit_top1 = 0
    n_rule_top1  = 0
    n_logit_eq_rule = 0
    n_total = 0

    score_term_means = {
        'w_mode_sm':  [],   # mode softmax max contribution
        'progress':   [],   # raw final x-disp
        'collision':  [],
        'drivable':   [],
        'comfort':    [],
    }

    rs = model.rule_selector  # type: ignore

    with torch.no_grad():
        for batch in loader:
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
            B = mode_logits.size(0)
            mode_label = batch['mode_label']

            logit_argmax = mode_logits.argmax(dim=-1)             # (B,)
            n_logit_top1 += (logit_argmax == mode_label).sum().item()
            n_rule_top1  += (best_idx     == mode_label).sum().item()
            n_logit_eq_rule += (logit_argmax == best_idx).sum().item()
            n_total += B

            # Per-term magnitudes (mirroring RuleSelector.forward math)
            mode_scores = F.softmax(mode_logits, dim=-1)                         # (B,M)
            score_term_means['w_mode_sm'].append(mode_scores.max(dim=-1).values.mean().item() * rs.w_mode)
            xy = all_trajs[..., :2]
            progress = all_trajs[:, :, -1, 0]                                    # (B,M)
            score_term_means['progress'].append((progress.abs().max(dim=-1).values.mean().item()) * rs.w_progress)
            # collision/drivable use the same internals as RuleSelector — shortcut: just bound
            score_term_means['collision'].append(rs.w_collision)  # max |term| ≈ 1
            score_term_means['drivable'].append(rs.w_drivable)
            vel  = xy[:, :, 1:] - xy[:, :, :-1]
            acc  = vel[:, :, 1:] - vel[:, :, :-1]
            jerk = acc[:, :, 1:] - acc[:, :, :-1]
            comfort = -jerk.norm(dim=-1).mean(dim=-1)                            # (B,M)
            score_term_means['comfort'].append(comfort.abs().max(dim=-1).values.mean().item() * rs.w_comfort)

    print()
    print(f"  selector_top1  (logits.argmax == mode_label): {n_logit_top1/n_total*100:.2f}%  ({n_logit_top1}/{n_total})")
    print(f"  rule_top1      (best_idx     == mode_label): {n_rule_top1 /n_total*100:.2f}%  ({n_rule_top1 }/{n_total})")
    print(f"  changed by rule (logit_argmax != best_idx):  {(1 - n_logit_eq_rule/n_total)*100:.2f}%")
    print()
    print(f"  Score term magnitudes (per-batch maxima, weighted):")
    for k, vals in score_term_means.items():
        print(f"    {k:<14} ≈ {np.mean(vals):.3f}")
    print()
    print(f"  RuleSelector weights:  w_mode={rs.w_mode}, w_progress={rs.w_progress}, "
          f"w_collision={rs.w_collision}, w_drivable={rs.w_drivable}, w_comfort={rs.w_comfort}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--cache', default='checkpoints/stage_cache_val14.pt')
    p.add_argument('--n_eval', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=32)
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
