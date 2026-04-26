"""Parse nohup_stage_c.out and plot train/val L_total + sub-components vs step."""
import re
from pathlib import Path
import matplotlib.pyplot as plt

LOG = Path("/media/skr/storage/autoresearch/CarPlanner_Implementation/nohup_stage_c.out")
OUT = Path("/media/skr/storage/autoresearch/CarPlanner_Implementation/stage_c_loss.png")

BATCHES_PER_EPOCH = 225

step_re = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\[(\d+)/(\d+)\]\s+L_total=([\d.]+)\s+L_policy=([\d.]+)\s+L_value=([\d.]+)"
)
epoch_re = re.compile(
    r"^Epoch\s+(\d+)/\d+\s+train=([\d.]+)\s+L_policy=([-\d.]+)\s+L_value=([-\d.]+)\s+"
    r"L_entropy=([-\d.]+)\s+L_selector=([-\d.]+)\s+val=([\d.]+)"
)

step_x, step_total, step_policy, step_value = [], [], [], []
epoch_x, ep_train, ep_pol, ep_val_loss, ep_ent, ep_sel, ep_val = [], [], [], [], [], [], []

with LOG.open() as f:
    for line in f:
        m = step_re.search(line)
        if m and line.lstrip().startswith("Epoch"):
            ep, b, _, lt, lp, lv = m.groups()
            ep, b = int(ep), int(b)
            step_x.append((ep - 1) * BATCHES_PER_EPOCH + b)
            step_total.append(float(lt))
            step_policy.append(float(lp))
            step_value.append(float(lv))
            continue
        m = epoch_re.match(line)
        if m:
            ep, tr, lp, lv, le, ls, va = m.groups()
            epoch_x.append(int(ep) * BATCHES_PER_EPOCH)
            ep_train.append(float(tr))
            ep_pol.append(float(lp))
            ep_val_loss.append(float(lv))
            ep_ent.append(float(le))
            ep_sel.append(float(ls))
            ep_val.append(float(va))

print(f"Parsed {len(step_x)} step points and {len(epoch_x)} epoch points")

fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
fig.suptitle("Stage C — RL Fine-tuning Losses vs Step", fontsize=13)


def add_epoch_axis(ax):
    ax.secondary_xaxis(
        "top",
        functions=(lambda s: s / BATCHES_PER_EPOCH, lambda e: e * BATCHES_PER_EPOCH),
    ).set_xlabel("Epoch")


# (0,0) Total loss linear
ax = axes[0, 0]
ax.plot(step_x, step_total, color="tab:blue", lw=0.7, alpha=0.5, label="train (per-batch)")
ax.plot(epoch_x, ep_train, "o-", color="tab:blue", lw=1.6, ms=4, label="train (epoch mean)")
ax.plot(epoch_x, ep_val, "s-", color="tab:red", lw=1.6, ms=4, label="val")
ax.set_title("L_total")
ax.set_ylabel("L_total")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=8)
add_epoch_axis(ax)

# (0,1) Total loss log
ax = axes[0, 1]
ax.plot(step_x, step_total, color="tab:blue", lw=0.7, alpha=0.5, label="train (per-batch)")
ax.plot(epoch_x, ep_train, "o-", color="tab:blue", lw=1.6, ms=4, label="train (epoch mean)")
ax.plot(epoch_x, ep_val, "s-", color="tab:red", lw=1.6, ms=4, label="val")
ax.set_yscale("log")
ax.set_title("L_total (log scale — spike visible)")
ax.set_ylabel("L_total")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", fontsize=8)
add_epoch_axis(ax)

# (1,0) Policy
ax = axes[1, 0]
ax.plot(step_x, step_policy, color="tab:green", lw=0.7, alpha=0.5, label="train (per-batch)")
ax.plot(epoch_x, ep_pol, "o-", color="tab:green", lw=1.6, ms=4, label="train (epoch mean)")
ax.set_yscale("log")
ax.set_title("L_policy (RL surrogate)")
ax.set_ylabel("L_policy")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", fontsize=8)

# (1,1) Value
ax = axes[1, 1]
ax.plot(step_x, step_value, color="tab:purple", lw=0.7, alpha=0.5, label="train (per-batch)")
ax.plot(epoch_x, ep_val_loss, "o-", color="tab:purple", lw=1.6, ms=4, label="train (epoch mean)")
ax.set_title("L_value (critic MSE)")
ax.set_ylabel("L_value")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=8)

# (2,0) Entropy
ax = axes[2, 0]
ax.plot(epoch_x, ep_ent, "o-", color="tab:orange", lw=1.6, ms=4, label="train (epoch mean)")
ax.set_title("L_entropy (negative = larger entropy)")
ax.set_ylabel("L_entropy")
ax.set_xlabel(f"Step (batches; {BATCHES_PER_EPOCH}/epoch)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=8)

# (2,1) Selector
ax = axes[2, 1]
ax.plot(epoch_x, ep_sel, "o-", color="tab:brown", lw=1.6, ms=4, label="train (epoch mean)")
ax.set_title("L_selector (mode-selector CE)")
ax.set_ylabel("L_selector")
ax.set_xlabel(f"Step (batches; {BATCHES_PER_EPOCH}/epoch)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT, dpi=140)
print(f"Saved {OUT}")

best_val = min(ep_val)
best_ep = epoch_x[ep_val.index(best_val)] // BATCHES_PER_EPOCH
print(f"Best val L_total = {best_val:.4f} at epoch {best_ep}")
print(f"Final (epoch 50): train={ep_train[-1]:.4f}  val={ep_val[-1]:.4f}  "
      f"L_policy={ep_pol[-1]:.4f}  L_value={ep_val_loss[-1]:.4f}  "
      f"L_entropy={ep_ent[-1]:.4f}  L_selector={ep_sel[-1]:.4f}")
