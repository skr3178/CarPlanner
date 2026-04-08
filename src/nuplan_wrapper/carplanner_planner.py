"""
CarPlanner planner wrapper for nuPlan closed-loop simulation.

Implements nuPlan's AbstractPlanner interface so CarPlanner checkpoints
can be evaluated on the Test14-Random / Test14-Hard benchmarks.
"""

import math
import time
import numpy as np
import torch
from typing import List, Optional, Type

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config as cfg
from model import CarPlanner
from src.nuplan_wrapper.carplanner_feature_builder import CarPlannerFeatureBuilder


def _rotate_points(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 2D points by angle (radians)."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    return points @ rot.T


def _global_trajectory_to_states(
    global_trajectory: np.ndarray,
    ego_history: List[EgoState],
    future_horizon: float,
    step_interval: float = 0.1,
) -> List[EgoState]:
    """
    Convert a sequence of (x, y, heading) waypoints into EgoState objects
    by copying dynamic properties from the current ego state.
    """
    current_ego = ego_history[-1]
    states = []

    for i, wp in enumerate(global_trajectory):
        # Create a new EgoState with interpolated position/heading
        # but same velocity/acceleration structure as current
        time_point = current_ego.time_point + i * step_interval

        # Build a new rear_axle pose
        from nuplan.common.actor_state.state_representation import StateSE2
        rear_axle = StateSE2(wp[0], wp[1], wp[2])

        # Clone the ego state with updated pose
        new_state = current_ego.clone_and_update_rear_axle(
            rear_axle=rear_axle,
            time_point=time_point,
        )
        states.append(new_state)

    return states


class CarPlannerPlanner(AbstractPlanner):
    """
    nuPlan-compatible planner wrapper for CarPlanner.

    Supports evaluating checkpoints from:
        Stage A — TransitionModel only (debugging)
        Stage B — IL-trained full model
        Stage C — RL-fine-tuned full model (paper results)
    """

    requires_scenario: bool = False

    def __init__(
        self,
        checkpoint_path: str = None,
        stage: str = 'c',
        replan_interval: int = 1,
        use_gpu: bool = True,
    ):
        super().__init__()

        self.checkpoint_path = checkpoint_path
        self.stage = stage
        self._replan_interval = replan_interval
        self._device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        # Timing
        self._step_interval = 0.1
        self._last_plan_elapsed_step = replan_interval  # force plan at first step
        self._global_trajectory = None
        self._start_time = None

        # Model and feature builder (initialized in initialize())
        self.model: Optional[CarPlanner] = None
        self.feature_builder = CarPlannerFeatureBuilder()
        self._initialization: Optional[PlannerInitialization] = None

        # Runtime stats
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

    def name(self) -> str:
        return f"CarPlanner_Stage{self.stage.upper()}"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Load model checkpoint and prepare for simulation."""
        torch.set_grad_enabled(False)
        self._initialization = initialization

        if self.checkpoint_path is not None:
            self.model = self._load_model()
            self.model.eval()
            self.model.to(self._device)

    def _load_model(self) -> CarPlanner:
        """Load CarPlanner model from checkpoint, stage-aware."""
        model = CarPlanner()

        if self.stage == 'a':
            # Stage A: only transition model is trained
            model.load_transition_model(self.checkpoint_path, freeze=True)
        else:
            # Stage B or C: full model checkpoint
            ckpt = torch.load(self.checkpoint_path, map_location='cpu')

            if isinstance(ckpt, dict) and 'model' in ckpt:
                state_dict = ckpt['model']
            elif isinstance(ckpt, dict):
                state_dict = ckpt
            else:
                state_dict = ckpt

            # Handle 'policy_old' for Stage C (PPO keeps old policy)
            model.load_state_dict(state_dict, strict=False)

            if self.stage == 'c' and hasattr(model, 'policy_old'):
                model.policy_old.eval()

        return model

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """
        Main planning loop called at each simulation step.

        1. Convert nuPlan state → CarPlanner tensors
        2. Run forward_inference()
        3. Convert output trajectory → nuPlan InterpolatedTrajectory
        """
        self._start_time = time.perf_counter()
        ego_state = current_input.history.ego_states[-1]

        if self._last_plan_elapsed_step >= self._replan_interval:
            # ── Replan ────────────────────────────────────────────────────
            features = self.feature_builder.get_features_from_simulation(
                current_input, self._initialization
            )
            self._feature_building_runtimes.append(
                time.perf_counter() - self._start_time)

            # Add batch dimension
            agents_now = torch.from_numpy(features['agents_now']).unsqueeze(0).to(self._device)
            agents_mask = torch.from_numpy(features['agents_mask']).unsqueeze(0).to(self._device)
            map_lanes = torch.from_numpy(features['map_lanes']).unsqueeze(0).to(self._device)
            map_lanes_mask = torch.from_numpy(features['map_lanes_mask']).unsqueeze(0).to(self._device)
            agents_history = torch.from_numpy(features['agents_history']).unsqueeze(0).to(self._device)

            # Run inference
            with torch.no_grad():
                if self.stage == 'a':
                    # Stage A: no policy, just use transition model output
                    # (limited — mainly for debugging)
                    agent_futures = self.model.transition_model(
                        agents_history, agents_mask, map_lanes, map_lanes_mask)
                    # Use mean future position as crude trajectory
                    local_traj = agent_futures[0, :, 0, :2].cpu().numpy()  # (T, 2)
                    headings = np.zeros(cfg.T_FUTURE, dtype=np.float64)
                    local_traj = np.concatenate(
                        [local_traj, headings[:, None]], axis=1)  # (T, 3)
                else:
                    # Stage B/C: full model inference
                    _, _, best_traj, _ = self.model.forward_inference(
                        agents_now=agents_now,
                        agents_mask=agents_mask,
                        map_lanes=map_lanes,
                        map_lanes_mask=map_lanes_mask,
                        agents_history=agents_history,
                    )
                    local_traj = best_traj[0].cpu().numpy().astype(np.float64)  # (T, 3)

            # Convert local (ego-centric) trajectory to global coordinates
            self._global_trajectory = self._local_to_global(
                local_traj, ego_state)
            self._last_plan_elapsed_step = 0
        else:
            # Advance along pre-computed trajectory
            self._global_trajectory = self._global_trajectory[1:]

        # Build nuPlan trajectory
        trajectory = InterpolatedTrajectory(
            trajectory=_global_trajectory_to_states(
                global_trajectory=self._global_trajectory,
                ego_history=current_input.history.ego_states,
                future_horizon=len(self._global_trajectory) * self._step_interval,
                step_interval=self._step_interval,
            )
        )

        self._inference_runtimes.append(time.perf_counter() - self._start_time)
        self._last_plan_elapsed_step += 1

        return trajectory

    def _local_to_global(
        self, local_trajectory: np.ndarray, ego_state: EgoState
    ) -> np.ndarray:
        """
        Convert ego-centric trajectory (x, y, yaw) to global coordinates.
        Uses ego's rear axle as the reference point.
        """
        origin = ego_state.rear_axle.array  # (x, y)
        angle = ego_state.rear_axle.heading

        # Rotate local positions to global
        global_xy = _rotate_points(
            local_trajectory[..., :2].astype(np.float64), -angle) + origin
        global_heading = local_trajectory[..., 2] + angle

        return np.concatenate(
            [global_xy, global_heading[..., None]], axis=-1)

    def generate_planner_report(self, clear_stats: bool = True):
        """Return timing statistics."""
        from nuplan.planning.simulation.planner.planner_report import MLPlannerReport

        report = MLPlannerReport(
            compute_trajectory_runtimes=self._inference_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._feature_building_runtimes = []
            self._inference_runtimes = []

        return report
