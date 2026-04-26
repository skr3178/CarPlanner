# CarPlanner

## Closed-loop test14-random — Paper vs Ours

### Stage B — paper IL-best vs ours

| Metric | Paper IL-best | Ours (Stage-B-seed) | Δ |
|---|---|---|---|
| **CLS-NR** | **93.41** | **17.00** | **−76.41** |
| S-CR (no-collision) | 98.85 | 35.1 | −63.75 |
| S-Area (drivable) | 98.85 | 29.1 | −69.75 |
| S-PR (progress) | 93.87 | 67.8 | −26.07 |
| S-Comfort | 96.15 | 43.8 | −52.35 |
| L_selector (open-loop, val14) | 1.04 | 2.31 | +1.27 |
| L_generator (open-loop, val14) | 174.3 | 12.6 | (different scale) |

### Stage C — paper RL-best vs ours

| Metric | Paper RL-best | Ours (Stage-C) | Δ |
|---|---|---|---|
| **CLS-NR** | **94.07** | **0.00** | **−94.07** |
| S-CR (no-collision) | 99.22 | 25.6 | −73.62 |
| S-Area (drivable) | 99.22 | 42.6 | −56.62 |
| S-PR (progress) | 95.06 | 8.4 | −86.66 |
| S-Comfort | 91.09 | 8.5 | −82.59 |
| L_selector (open-loop, val14) | 1.03 | 2.47 | +1.44 |
| L_generator (open-loop, val14) | 1624.5 | 14.4 | (different scale) |
