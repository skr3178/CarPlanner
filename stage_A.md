# Plan: Stage A — Transition Model Pre-training

## Context
CarPlanner has three stages: A (transition model), B (IL), C (RL). Stage B is working.
Stage A must be trained and frozen before Stage C (RL) can use it as a simulator.
The TransitionModel class already exists in model.py but has never been trained —
there is no training loop, no loss function, and no checkpoint for it.

## What needs to be built

### 1. Loss function — Eq 5
File: new `train_stage_a.py`

```python
def compute_transition_loss(pred, gt, mask):
    # pred: (B, T, N, Da), gt: (B, T, N, Da), mask: (B, N)
    diff = (pred - gt).abs().sum(dim=-1)        # (B, T, N) — L1 per agent
    mask_t = mask.unsqueeze(1).expand_as(diff)  # (B, T, N)
    return (diff * mask_t).sum() / (mask_t.sum() + 1e-6)
```

Mask is critical: only compute loss for valid agents (mask=1), not padding slots.
Paper Eq 5 sums over N agents and averages over T — masking ensures padding agents
don't dilute the gradient.

### 2. Training script — `train_stage_a.py`
Separate script (keeps Stage A / Stage B concerns cleanly separated).

Structure mirrors train.py:
- Load data with `make_dataloader(split)` — same dataloader, no changes needed
- Instantiate only `TransitionModel` (not full `CarPlanner`)
- Forward: `agents_pred = model(agents_now, agents_mask)` → `(B, T, N, Da)`
- Loss: `L_tm = compute_transition_loss(agents_pred, agents_seq, agents_mask)`
- AdamW + ReduceLROnPlateau (same hyperparams as Stage B from config.py)
- Save checkpoint: `checkpoints/stage_a_best.pt` containing `{'model': state_dict, 'epoch': ..., 'loss': ...}`
- CLI args: `--split`, `--epochs`, `--batch_size`, `--resume`

### 3. Checkpoint loading in CarPlanner
File: `model.py` — add a `load_transition_model()` method to `CarPlanner`:

```python
def load_transition_model(self, ckpt_path: str, freeze: bool = True):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    self.transition_model.load_state_dict(ckpt['model'])
    if freeze:
        for p in self.transition_model.parameters():
            p.requires_grad = False
        self.transition_model.eval()
```

This is called in `train.py` before Stage B (and will be called in Stage C).

### 4. Wire into train.py
Add `--transition_ckpt` arg to `train.py` so Stage B training can load the frozen
transition model at startup (currently it's never used — this makes it explicit).

## Files to modify / create

| File | Change |
|------|--------|
| `train_stage_a.py` | CREATE — full Stage A training script |
| `model.py` | ADD `load_transition_model()` method to `CarPlanner` (~10 lines) |
| `train.py` | ADD `--transition_ckpt` arg + call `load_transition_model()` if provided |

**No changes to:** `config.py`, `data_loader.py`, `evaluate.py`

## Key design decisions

- **Separate script** (not integrated into train.py): Stage A is conceptually independent and trains on different targets (agent futures, not ego trajectory). Keeps both scripts readable.
- **Mask the loss**: Eq 5 sums over N agents; without masking, padding zeros in `agents_seq` would provide false supervision signal.
- **Only TransitionModel instantiated**: Don't construct full CarPlanner — avoids loading unnecessary submodules and keeps GPU memory low during Stage A.
- **Freeze on load**: `load_transition_model(freeze=True)` sets `requires_grad=False` and `.eval()` mode — the paper explicitly states it's frozen during stages 2–3.

## Verification
1. Run: `python train_stage_a.py --split mini --epochs 5`
   - Expect: L_tm decreasing each epoch, no NaN
2. Check checkpoint saved at `checkpoints/stage_a_best.pt`
3. Run Stage B with transition model loaded:
   `python train.py --split mini --epochs 1 --transition_ckpt checkpoints/stage_a_best.pt`
   - Expect: loads without error, Stage B loss unchanged (transition model frozen, not used in IL forward pass directly)



  Inputs to TransitionModel:
  - agents_now — (B, N=20, Da=10) — agent states at t=0 in ego frame (x, y,
  heading, vx, vy, box_w, box_l, box_h, time_step, category)
  - agents_mask — (B, N=20) — validity mask (1=valid agent, 0=padding)

  Output from TransitionModel:
  - agents_pred — (B, T=8, N=20, Da=10) — predicted future agent states for
  all 8 steps

  Ground truth label (for loss only):
  - agents_seq — (B, T=8, N=20, Da=10) — GT future agent states from the
  dataset (already provided by the dataloader)

  ---
  What the loss (Eq 5) computes:

  L_tm = masked L1 between agents_pred and agents_seq
       = mean over (B, T, valid N) of ||pred - gt||_1

  The mask ensures padding agents (slots where no real agent exists) don't
  contribute to the loss — only the valid agent slots (mask=1) are counted.

#claude
claude --resume c9c246ec-6d4a-48b2-9aa6-bae8fc225246

#claude-glm
claude --resume 5ef859bf-e63b-4166-acd9-096712d9ad25