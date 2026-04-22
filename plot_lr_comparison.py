"""Plot LR schedule and val loss comparison across all Stage A runs."""
import re
import matplotlib.pyplot as plt
import numpy as np

logs = {
    'Run 0: original mini+boston\n(train_all, 50 ep)': 'checkpoints/stage_a_20260422_112814/stage_a_train_all.log',
    'Run 1: 2.66M balanced, p=0\n(112814, 50 ep)': 'logs/train_stage_a_20260422_112814.log',
    'Run 2: 3.45M balanced, p=0\n(122139, killed @36)': 'logs/train_stage_a_20260422_122139.log',
    'Run 3: 3.45M balanced, p=5\n(123906, in progress)': 'logs/train_stage_a_20260422_123906.log',
}

pattern = re.compile(
    r'Epoch\s+(\d+)/\d+\s+train=([\d.]+)\s+val=([\d.]+)\s+lr=([\d.eE+-]+)'
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
colors = ['#9b59b6', '#e74c3c', '#3498db', '#2ecc71']

all_data = {}
for (label, path), color in zip(logs.items(), colors):
    epochs, lrs, train_losses, val_losses = [], [], [], []
    try:
        with open(path) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    epochs.append(int(m.group(1)))
                    train_losses.append(float(m.group(2)))
                    val_losses.append(float(m.group(3)))
                    lrs.append(float(m.group(4)))
    except FileNotFoundError:
        continue
    if not epochs:
        continue
    all_data[label] = (epochs, lrs, train_losses, val_losses, color)

# Plot 1: LR schedule (log scale)
ax = axes[0]
for label, (epochs, lrs, _, _, color) in all_data.items():
    ax.semilogy(epochs, lrs, 'o-', color=color, label=label, markersize=3, linewidth=1.5)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Learning Rate (log scale)', fontsize=11)
ax.set_title('LR Schedule Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 52)

# Add annotations for decay points
for label, (epochs, lrs, _, _, color) in all_data.items():
    prev_lr = lrs[0]
    for i, (e, lr) in enumerate(zip(epochs, lrs)):
        if lr < prev_lr * 0.5:
            ax.annotate(f'→{lr:.0e}', (e, lr), fontsize=6, color=color,
                       textcoords='offset points', xytext=(5, 5))
        prev_lr = lr

# Plot 2: Val loss
ax = axes[1]
for label, (epochs, _, _, val_losses, color) in all_data.items():
    ax.plot(epochs, val_losses, 'o-', color=color, label=label, markersize=3, linewidth=1.5)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Validation Loss (L_tm)', fontsize=11)
ax.set_title('Validation Loss', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 52)

# Plot 3: Val loss vs LR (shows how much improvement per LR level)
ax = axes[2]
for label, (epochs, lrs, _, val_losses, color) in all_data.items():
    ax.semilogx(lrs, val_losses, 'o-', color=color, label=label, markersize=3, linewidth=1.5)
    # Mark start and end
    ax.plot(lrs[0], val_losses[0], 's', color=color, markersize=8, zorder=5)
    ax.plot(lrs[-1], val_losses[-1], '*', color=color, markersize=12, zorder=5)
ax.set_xlabel('Learning Rate (log scale)', fontsize=11)
ax.set_ylabel('Validation Loss (L_tm)', fontsize=11)
ax.set_title('Val Loss vs LR\n(□=start, ★=end)', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
ax.invert_xaxis()

plt.tight_layout()
plt.savefig('plots/stage_a_lr_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: plots/stage_a_lr_comparison.png")
plt.close()
