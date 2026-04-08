#!/usr/bin/env python3
"""
CarPlanner nuPlan closed-loop evaluation.

Runs CarPlanner checkpoints on nuPlan benchmarks (Test14-Random, Test14-Hard)
using nuPlan's simulation framework to compute CLS-NR, S-CR, S-Area, S-PR,
S-Comfort metrics.

Prerequisites:
    - nuPlan devkit installed (pip install -e . in nuplan-devkit/)
    - nuPlan dataset downloaded and extracted
    - CarPlanner checkpoint saved from training

Usage:
    # Evaluate Stage C on Test14-Random (non-reactive):
    python scripts/eval_nuplan.py \
        --checkpoint checkpoints/stage_c_best.pt \
        --stage c \
        --split test14-random

    # Evaluate Stage C on Test14-Hard:
    python scripts/eval_nuplan.py \
        --checkpoint checkpoints/stage_c_best.pt \
        --stage c \
        --split test14-hard

    # Evaluate Stage B for comparison:
    python scripts/eval_nuplan.py \
        --checkpoint checkpoints/best.pt \
        --stage b \
        --split test14-random

    # Quick test with single scenario:
    python scripts/eval_nuplan.py \
        --checkpoint checkpoints/stage_c_best.pt \
        --stage c \
        --split single_right_turn \
        --threads 1

Available splits (from planTF/config/scenario_filter/):
    test14-random   — 261 scenarios, 14 types
    test14-hard     — ~500 scenarios, harder cases
    val14           — 280 scenarios (20 per type)
    mini            — small subset for quick testing

Available challenges:
    closed_loop_nonreactive_agents  — log-replay agents (paper default)
    closed_loop_reactive_agents     — IDM-reactive agents
    open_loop_boxes                 — open-loop evaluation
"""

import argparse
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description='CarPlanner nuPlan closed-loop evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--checkpoint', required=True,
        help='Path to CarPlanner checkpoint (.pt file)')
    parser.add_argument(
        '--stage', default='c', choices=['a', 'b', 'c'],
        help='Training stage of the checkpoint (default: c)')
    parser.add_argument(
        '--split', default='test14-random',
        help='Scenario split to evaluate on (default: test14-random)')
    parser.add_argument(
        '--challenge', default='closed_loop_nonreactive_agents',
        choices=[
            'closed_loop_nonreactive_agents',
            'closed_loop_reactive_agents',
            'open_loop_boxes',
        ],
        help='Simulation challenge type (default: closed_loop_nonreactive_agents)')
    parser.add_argument(
        '--threads', type=int, default=20,
        help='Number of worker threads (default: 20)')
    parser.add_argument(
        '--output_dir', default=None,
        help='Output directory (default: nuplan_eval/{split}/)')
    parser.add_argument(
        '--gpu', action='store_true', default=True,
        help='Use GPU for inference (default: True)')
    parser.add_argument(
        '--verbose', action='store_true', default=True,
        help='Verbose output (default: True)')

    args = parser.parse_args()

    # Validate checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Set output directory
    output_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, 'nuplan_eval', f'{args.split}_stage{args.stage}')

    # Import nuPlan simulation runner
    try:
        from nuplan.planning.script.run_simulation import main as run_simulation
        from nuplan.planning.script.utils import set_up_common_builder
    except ImportError:
        print("ERROR: nuPlan devkit not installed.")
        print("Install it with:")
        print("  git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit")
        print("  pip install -e . && pip install -r requirements.txt")
        sys.exit(1)

    # Build simulation config
    # Using nuPlan's Hydra-based config system
    overrides = [
        f'+simulation={args.challenge}',
        f'scenario_builder=nuplan_challenge',
        f'scenario_filter={args.split}',
        f'worker=multi_node_thread_pool',
        f'worker.threads_per_node={args.threads}',
        f'experiment_uid=carplanner/stage{args.stage}/{args.split}',
        f'output_dir={output_dir}',
        'verbose=true',
        'run_metric=true',
    ]

    # Set environment for Hydra
    os.environ['PYTHONPATH'] = PROJECT_ROOT + ':' + os.environ.get('PYTHONPATH', '')

    # Add planTF config paths for scenario filter access
    plantf_config = os.path.join(PROJECT_ROOT, 'planTF', 'config')
    scenario_filter_dir = os.path.join(plantf_config, 'scenario_filter')

    print("=" * 60)
    print("CarPlanner nuPlan Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Stage:      {args.stage}")
    print(f"  Split:      {args.split}")
    print(f"  Challenge:  {args.challenge}")
    print(f"  Output:     {output_dir}")
    print(f"  Threads:    {args.threads}")
    print("=" * 60)

    # Run via Hydra compose API
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    # Use PlanTF's simulation config as base
    plantf_sim_config = os.path.join(plantf_config)

    with initialize_config_dir(config_dir=plantf_sim_config, version_base=None):
        cfg = compose(config_name='default_simulation', overrides=overrides)
        print("\nConfig resolved. Starting simulation...\n")
        run_simulation(cfg)


if __name__ == '__main__':
    main()
