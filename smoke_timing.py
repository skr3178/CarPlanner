"""Measure Stage B step time for one config at a time (to avoid memory fragmentation)."""
import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import config as cfg
from model import CarPlanner


def measure(cache_path, transition_ckpt, batch_size, pin_gpu, n_steps=10):
    device = torch.device('cuda')
    if pin_gpu:
        data = torch.load(cache_path, map_location=device)
    else:
        data = torch.load(cache_path, map_location='cpu')
    n_samples = data['n_samples']

    model = CarPlanner().to(device)
    model.load_transition_model(transition_ckpt, freeze=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR)

    def run_step(step):
        start = (step * batch_size) % (n_samples - batch_size)
        if pin_gpu:
            idx = torch.arange(start, start + batch_size, device=device)
            batch = {k: data[k][idx] for k in ['agents_now','agents_history','agents_mask','agents_seq',
                                                'gt_trajectory','mode_label','map_lanes','map_lanes_mask',
                                                'map_polygons','map_polygons_mask']}
        else:
            batch = {k: data[k][start:start+batch_size].to(device) for k in ['agents_now','agents_history','agents_mask','agents_seq',
                                                                              'gt_trajectory','mode_label','map_lanes','map_lanes_mask',
                                                                              'map_polygons','map_polygons_mask']}
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, side, pred = model.forward_train(
                batch['agents_now'], batch['agents_mask'], batch['agents_seq'],
                batch['gt_trajectory'], batch['mode_label'],
                map_lanes=batch['map_lanes'], map_lanes_mask=batch['map_lanes_mask'],
                agents_history=batch['agents_history'],
                map_polygons=batch['map_polygons'], map_polygons_mask=batch['map_polygons_mask'])
            loss = F.cross_entropy(logits, batch['mode_label']) + (side-batch['gt_trajectory']).abs().sum(-1).mean() + (pred-batch['gt_trajectory']).abs().sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Warmup
    for s in range(3):
        run_step(s)
    torch.cuda.synchronize()

    t0 = time.time()
    for s in range(n_steps):
        run_step(s)
    torch.cuda.synchronize()
    return (time.time() - t0) / n_steps


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int, required=True)
    p.add_argument('--pin_gpu', action='store_true')
    args = p.parse_args()

    cache = os.path.join(cfg.CHECKPOINT_DIR, "stage_cache_train_4city_balanced.pt")
    tck = os.path.join(cfg.CHECKPOINT_DIR, "stage_a_best.pt")

    step_s = measure(cache, tck, args.batch_size, args.pin_gpu)
    n_train = int(176054 * 0.9)
    samples_s = args.batch_size / step_s
    epoch_s = n_train / samples_s
    total_h = 50 * epoch_s / 3600
    print(f"RESULT bs={args.batch_size} pin={args.pin_gpu} step={step_s:.3f}s samples/s={samples_s:.0f} epoch={epoch_s:.0f}s total_50ep={total_h:.2f}h")
