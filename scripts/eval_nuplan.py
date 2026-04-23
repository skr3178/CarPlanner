#!/usr/bin/env python3
"""
CarPlanner closed-loop evaluation via nuPlan simulation framework.

Runs the CarPlanner planner through nuPlan's official simulation pipeline,
producing CLS-NR, S-CR, S-Area, S-PR, S-Comfort metrics.

This script invokes planTF/run_simulation.py with the correct Hydra overrides
to use our CarPlannerPlanner wrapper.

Usage:
    # Stage B on test14-random (non-reactive):
    python scripts/eval_nuplan.py \
        --checkpoint checkpoints/stage_b_best.pt \
        --split test14-random

    # Stage B on val14:
    python scripts/eval_nuplan.py \
        --checkpoint checkpoints/stage_b_best.pt \
        --split val14

    # Reactive agents:
    python scripts/eval_nuplan.py \
        --checkpoint checkpoints/stage_b_best.pt \
        --split test14-random \
        --challenge closed_loop_reactive_agents

    # Quick smoke test (single scenario type):
    python scripts/eval_nuplan.py \
        --checkpoint checkpoints/stage_b_best.pt \
        --split single_right_turn \
        --threads 1

Available splits: test14-random, test14-hard, val14, mini, single_right_turn
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description='CarPlanner nuPlan closed-loop evaluation',
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to CarPlanner checkpoint (.pt)')
    parser.add_argument('--stage', default='b', choices=['a', 'b', 'c'],
                        help='Training stage (default: b)')
    parser.add_argument('--split', default='test14-random',
                        help='Scenario filter (default: test14-random)')
    parser.add_argument('--challenge', default='closed_loop_nonreactive_agents',
                        choices=['closed_loop_nonreactive_agents',
                                 'closed_loop_reactive_agents',
                                 'open_loop_boxes'],
                        help='Simulation type (default: closed_loop_nonreactive_agents)')
    parser.add_argument('--threads', type=int, default=24,
                        help='Worker threads (default: 24)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: auto)')
    args = parser.parse_args()

    checkpoint_abs = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_abs):
        print(f"ERROR: Checkpoint not found: {checkpoint_abs}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, 'nuplan_eval', f'{args.split}_stage{args.stage}')

    # Environment setup
    env = os.environ.copy()
    env['CARPLANNER_CHECKPOINT'] = checkpoint_abs

    maps_root = os.path.join(PROJECT_ROOT, 'paper/dataset/maps')
    env.setdefault('NUPLAN_MAPS_ROOT', maps_root)
    env.setdefault('NUPLAN_DATA_ROOT', os.path.join(PROJECT_ROOT, 'paper/dataset'))

    # PYTHONPATH must include project root so Hydra can instantiate carplanner_planner
    pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = PROJECT_ROOT + (':' + pythonpath if pythonpath else '')

    db_path = '/home/skr/nuplan_cities/val/data/cache/val'

    overrides = [
        f'+simulation={args.challenge}',
        'scenario_builder=nuplan_challenge',
        f'scenario_builder.db_files={db_path}',
        f'scenario_filter={args.split}',
        'planner=carplanner',
        f'planner.stage={args.stage}',
        f'worker=sequential',
        f'output_dir={output_dir}',
        'verbose=true',
        'run_metric=true',
    ]

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, 'planTF', 'run_simulation.py'),
    ] + overrides

    print("=" * 60)
    print("CarPlanner nuPlan Closed-Loop Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint_abs}")
    print(f"  Stage:      {args.stage}")
    print(f"  Split:      {args.split}")
    print(f"  Challenge:  {args.challenge}")
    print(f"  Output:     {output_dir}")
    print(f"  DB path:    {db_path}")
    print("=" * 60)
    print(f"\nCommand:\n  {' '.join(cmd)}\n")

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
