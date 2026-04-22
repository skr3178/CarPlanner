# Training Run Notes

## Stage B — Batch Size & Training Time Benchmarks

**Stage**: Stage B (IL training: ModeSelector + AutoregressivePolicy)
**Dataset**: `stage_cache_train_4city_balanced.pt` — 176,054 samples, 52 scenario types, 10.2 GB
  - Boston: 36,550 + Pittsburgh: 41,045 + Singapore: 38,346 + Vegas: 60,113
  - Train split (90%): 158,448 samples
**Stage A checkpoint**: `checkpoints/stage_a_best.pt` (epoch 50, val L_tm=9.6350, 3.45M params)
**Hardware**: RTX 3060 12 GB (12.5 GB total)
**Model**: CarPlanner, 6,746,387 total params (3,298,787 trainable in Stage B)

### Max Batch Size Limits

| Mode | Max batch size | Peak GPU memory |
|---|---|---|
| `--pin_gpu` (cache on GPU) | **160** | 11.91 GB |
| No `--pin_gpu` (cache on CPU) | **1024** | 10.48 GB |

### Step Time Measurements (10 steps after 3 warmup)

| Config | BS | Step time | Samples/s | Epoch time | 50 epochs |
|---|---|---|---|---|---|
| `--pin_gpu` | 64 | 0.347 s | 185 | 14.3 min | **11.92 h** |
| `--pin_gpu` | 128 | 0.687 s | 186 | 14.2 min | **11.82 h** |
| `--pin_gpu` | 160 (max) | 0.861 s | 186 | 14.2 min | **11.85 h** |
| no pin_gpu | 256 | 1.334 s | 192 | 13.8 min | **11.47 h** |
| no pin_gpu | 512 | 2.689 s | 190 | 13.9 min | **11.56 h** |
| no pin_gpu | 1024 (max) | 5.377 s | 190 | 13.9 min | **11.56 h** |

### Key Finding

**Throughput is GPU-bound, identical across all configs (~185-192 samples/s).**
- Neither pin_gpu nor batch size affects overall training speed.
- Total 50-epoch time is ~11.5-12 hours regardless of config.
- Batch size only affects gradient statistics, not throughput.

### Recommendation

Use **`--batch_size 128 --pin_gpu`** to match the paper's effective batch size (paper: 64 per GPU × 2 GPUs = 128).

### Command

```bash
python train_stage_b.py --split train_4city_balanced --batch_size 128 --pin_gpu \
    --transition_ckpt checkpoints/stage_a_best.pt --epochs 50
```

### Comparison with 3-City Run

The previous Stage B run on 3-city (115,941 samples, pin_gpu=True, bs=64) showed:
- ~550s/epoch → 14.2 min/epoch (matches current throughput ~185 samples/s × 104,347 samples = 564s)
- 4-city will take ~52% longer per epoch due to more samples.
