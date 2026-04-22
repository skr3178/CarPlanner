"""
Smoke test for Stage B training batch sizes on 4-city cache.
Tests max batch size with pin_gpu ON and OFF.
"""
import os
import sys
import time
import torch
import torch.nn.functional as F
import argparse

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import config as cfg
from model import CarPlanner


def try_batch_size(cache_path, transition_ckpt, batch_size, pin_gpu):
    """Try a batch size; return True if fits, False if OOM."""
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        if pin_gpu:
            data = torch.load(cache_path, map_location=device)
            n_samples = data['n_samples']
            gpu_cache = {
                'agents_now':          data['agents_now'],
                'agents_history':      data['agents_history'],
                'agents_mask':         data['agents_mask'],
                'agents_seq':          data['agents_seq'],
                'gt_trajectory':       data['gt_trajectory'],
                'mode_label':          data['mode_label'],
                'map_lanes':           data['map_lanes'],
                'map_lanes_mask':      data['map_lanes_mask'],
                'map_polygons':        data['map_polygons'],
                'map_polygons_mask':   data['map_polygons_mask'],
            }
            del data
            cache_gb = torch.cuda.memory_allocated() / 1e9
            print(f"    Cache loaded on GPU: {cache_gb:.2f} GB")
        else:
            data = torch.load(cache_path, map_location='cpu')
            n_samples = data['n_samples']
            gpu_cache = data
            cache_gb = torch.cuda.memory_allocated() / 1e9
            print(f"    Cache on CPU (GPU used so far: {cache_gb:.2f} GB)")

        model = CarPlanner().to(device)
        if transition_ckpt:
            model.load_transition_model(transition_ckpt, freeze=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR)

        # Get a batch
        idx = torch.arange(batch_size)
        if pin_gpu:
            idx_d = idx.to(device)
            agents_now = gpu_cache['agents_now'][idx_d]
            agents_history = gpu_cache['agents_history'][idx_d]
            agents_mask = gpu_cache['agents_mask'][idx_d]
            agents_seq = gpu_cache['agents_seq'][idx_d]
            gt_traj = gpu_cache['gt_trajectory'][idx_d]
            mode_label = gpu_cache['mode_label'][idx_d]
            map_lanes = gpu_cache['map_lanes'][idx_d]
            map_lanes_mask = gpu_cache['map_lanes_mask'][idx_d]
            map_polygons = gpu_cache['map_polygons'][idx_d]
            map_polygons_mask = gpu_cache['map_polygons_mask'][idx_d]
        else:
            agents_now = gpu_cache['agents_now'][:batch_size].to(device)
            agents_history = gpu_cache['agents_history'][:batch_size].to(device)
            agents_mask = gpu_cache['agents_mask'][:batch_size].to(device)
            agents_seq = gpu_cache['agents_seq'][:batch_size].to(device)
            gt_traj = gpu_cache['gt_trajectory'][:batch_size].to(device)
            mode_label = gpu_cache['mode_label'][:batch_size].to(device)
            map_lanes = gpu_cache['map_lanes'][:batch_size].to(device)
            map_lanes_mask = gpu_cache['map_lanes_mask'][:batch_size].to(device)
            map_polygons = gpu_cache['map_polygons'][:batch_size].to(device)
            map_polygons_mask = gpu_cache['map_polygons_mask'][:batch_size].to(device)

        # Forward + backward (IL mode)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            mode_logits, side_traj, pred_traj = model.forward_train(
                agents_now, agents_mask, agents_seq, gt_traj, mode_label,
                map_lanes=map_lanes, map_lanes_mask=map_lanes_mask,
                agents_history=agents_history,
                map_polygons=map_polygons, map_polygons_mask=map_polygons_mask,
            )
            L_CE = F.cross_entropy(mode_logits, mode_label)
            L_side = (side_traj - gt_traj).abs().sum(dim=-1).mean()
            L_gen = (pred_traj - gt_traj).abs().sum(dim=-1).mean()
            L_total = L_CE + L_side + L_gen

        optimizer.zero_grad()
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        return True, peak_gb, None

    except torch.cuda.OutOfMemoryError as e:
        return False, 0, str(e)[:80]
    except Exception as e:
        return False, 0, f"{type(e).__name__}: {str(e)[:80]}"


def main():
    cache_path = os.path.join(cfg.CHECKPOINT_DIR, "stage_cache_train_4city_balanced.pt")
    transition_ckpt = os.path.join(cfg.CHECKPOINT_DIR, "stage_a_best.pt")

    print(f"Cache: {cache_path}")
    print(f"Stage A: {transition_ckpt}")
    print(f"GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Test with pin_gpu=True (load whole cache to GPU)
    print("=" * 70)
    print("PIN_GPU = TRUE (cache on GPU)")
    print("=" * 70)
    for bs in [128, 160, 192, 224, 256]:
        print(f"\n  batch_size={bs}")
        ok, peak, err = try_batch_size(cache_path, transition_ckpt, bs, pin_gpu=True)
        if ok:
            print(f"    OK  peak={peak:.2f} GB")
        else:
            print(f"    FAIL  {err}")
            break

    # Clear GPU between tests
    torch.cuda.empty_cache()

    # Test with pin_gpu=False (cache on CPU)
    print()
    print("=" * 70)
    print("PIN_GPU = FALSE (cache on CPU)")
    print("=" * 70)
    for bs in [1024, 1280, 1536, 1792, 2048]:
        print(f"\n  batch_size={bs}")
        ok, peak, err = try_batch_size(cache_path, transition_ckpt, bs, pin_gpu=False)
        if ok:
            print(f"    OK  peak={peak:.2f} GB")
        else:
            print(f"    FAIL  {err}")
            break


if __name__ == '__main__':
    main()
