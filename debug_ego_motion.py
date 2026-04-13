"""Quick diagnostic: is the ego vehicle moving during closed-loop eval?"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import config as cfg
from model import CarPlanner
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
from eval_closedloop_gpu import prestack, build_batch_inputs, ego_to_world_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load cache + model
data = torch.load('checkpoints/test14_random_cache.pt', map_location='cpu')
scenarios = data['scenarios'][:8]  # just 8 scenarios
print(f"Loaded {len(scenarios)} scenarios")

model = CarPlanner().to(device)
ckpt = torch.load('checkpoints/stage_c_best.pt', map_location=device)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()
print(f"Model loaded\n")

batch = prestack(scenarios)
B = batch['ego_gt'].size(0)
ego_states = batch['ego_gt'][:, 0, :3].clone()  # (B, 3) x,y,yaw

print("=== GT ego trajectory (first 10 steps) ===")
for b in range(min(3, B)):
    gt = batch['ego_gt'][b, :10, :2]
    dists = torch.norm(gt[1:] - gt[:-1], dim=-1)
    total = torch.norm(gt[-1] - gt[0])
    print(f"  Scenario {b}: start=({gt[0,0]:.1f}, {gt[0,1]:.1f})  "
          f"step_dists={dists[:5].tolist()}  total_10step={total:.2f}m")

print("\n=== Model predictions (single replan) ===")
with torch.no_grad():
    inp = build_batch_inputs(batch, 0, ego_states, device)

    # Check input statistics
    print(f"\n  agents_now stats: mean={inp['agents_now'].mean():.4f}, std={inp['agents_now'].std():.4f}")
    print(f"  agents_mask sum per sample: {inp['agents_mask'].sum(dim=1).tolist()}")
    print(f"  map_lanes stats: mean={inp['map_lanes'].mean():.4f}, std={inp['map_lanes'].std():.4f}")
    print(f"  map_lanes_mask sum per sample: {inp['map_lanes_mask'].sum(dim=1).tolist()}")

    _, _, best_traj, _ = model.forward_inference(
        agents_now=inp['agents_now'],
        agents_mask=inp['agents_mask'],
        map_lanes=inp['map_lanes'],
        map_lanes_mask=inp['map_lanes_mask'],
        agents_history=inp['agents_history'],
    )
    # best_traj: (B, T_FUTURE, 3) in ego frame
    print(f"\n  best_traj shape: {best_traj.shape}")
    print(f"  best_traj stats: mean={best_traj.mean():.4f}, std={best_traj.std():.4f}")
    print(f"  best_traj range: [{best_traj.min():.4f}, {best_traj.max():.4f}]")

    for b in range(min(3, B)):
        traj = best_traj[b]  # (T_FUTURE, 3)
        print(f"\n  Scenario {b} predicted traj (ego frame):")
        for t in range(traj.shape[0]):
            print(f"    t={t}: x={traj[t,0]:.4f}, y={traj[t,1]:.4f}, yaw={traj[t,2]:.4f}")

        # Total displacement
        disp = torch.norm(traj[-1, :2] - traj[0, :2])
        print(f"    Total displacement: {disp:.4f}m")

    # Transform to world
    pred_world = ego_to_world_batch(best_traj[:, :, :2].cpu(), ego_states)
    print(f"\n  World-frame displacement per scenario:")
    for b in range(min(3, B)):
        d = torch.norm(pred_world[b, -1] - pred_world[b, 0])
        print(f"    Scenario {b}: {d:.4f}m")
