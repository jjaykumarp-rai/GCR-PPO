# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections import deque
from math import gcd
from typing import Deque, Tuple
from collections.abc import Sequence

import torch
import torch.nn.functional as F
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaacsim.core.utils.stage as stage_utils

from .replicate_g1_throw_env_cfg import ReplicateG1ThrowEnvCfg
from .alpha_utils import (
    TORSO_JOINTS,
    RIGHT_ARM_JOINTS,
    LEFT_ARM_JOINTS,
    FINGER_JOINTS,
)

# -----------------------------------------------------------------------------
# Canonical joint ordering conventions
#
# We maintain a canonical ordering so that:
# - action decoding is consistent across robot versions
# - joint index mapping is robust to USD joint ordering differences
# -----------------------------------------------------------------------------
CANONICAL_MAIN_JOINTS = TORSO_JOINTS + RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS
CANONICAL_FINGER_JOINTS = FINGER_JOINTS
CANONICAL_JOINTS = CANONICAL_MAIN_JOINTS + CANONICAL_FINGER_JOINTS

# Target pose for the finger joints to represent a "closed hand" grasp posture.
# Ordering must match CANONICAL_FINGER_JOINTS.
CLOSED_FINGER_TARGETS = (
    0.85, 0.85, 0.85,
    0.65, 0.65, 0.65,
    0.85, 0.85, 0.85,
    0.65, 0.65, 0.65,
)


class ReplicateG1ThrowEnv(DirectRLEnv):
    """Direct RL environment for Alpha/G1-style whole-body throwing.

    This environment is a "rectified" variant designed to make the *hit target*
    objective consistent and learnable.

    Core design principles reflected in this implementation:
    1) **Single source of truth for target position**
       - Target world position is always: `target_positions + env_origins`.
       - Avoids subtle coordinate drift caused by root-state offsets.

    2) **Position-action control semantics**
       - Main joint actions are interpreted as *position deltas around default pose*.
       - More stable than integrating velocities when training position-target policies.

    3) **Reward anchored to real outcome**
       - Adds a reward component based on the ball’s (predicted) landing error,
         so dense shaping stays tied to the actual objective.

    Where to look for what:
    - Scene/asset creation: `_setup_scene`
    - Joint name → DOF index mapping: `_build_alpha_joint_indices`
    - Action decoding: `_pre_physics_step` + `_apply_action`
    - Observations: `_get_observations`
    - Rewards: `_get_rewards`
    - Termination logic: `_get_dones`
    - Resets + command sampling: `_reset_idx`, `_sample_throwing_commands`, `calculate_target_offset`

    Notes for KT:
    - The environment runs large-scale vectorized rollouts (up to 4096 envs).
      Anything per-env should be batched/tensorized.
    - Rendering assets (target) are only spawned when GUI/rendering is enabled.
    """

    cfg: ReplicateG1ThrowEnvCfg

    def __init__(self, cfg: ReplicateG1ThrowEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment and allocate persistent buffers.

        Args:
            cfg: Environment config dataclass (`ReplicateG1ThrowEnvCfg`).
            render_mode: Gymnasium render mode; forwarded to base env.
            **kwargs: Extra kwargs forwarded to the `DirectRLEnv` constructor.
        """
        num_envs = getattr(cfg.scene, "num_envs", 0)
        self._ball_trace_color_ints: dict[int, int] = self._initialize_ball_trace_color_ints(num_envs)
        super().__init__(cfg, render_mode, **kwargs)

        # ---------------------------------------------------------------------
        # Canonical DOF mapping: build index tensors used throughout the env
        # ---------------------------------------------------------------------
        self._build_alpha_joint_indices()
        self.main_joint_indices = self._main_joint_ids
        self.finger_joint_indices = self._finger_joint_ids

        # Closed-hand grasp posture (per-env expanded when used)
        self._closed_finger_pose = torch.tensor(CLOSED_FINGER_TARGETS, device=self.device)

        # Placeholder for desired joint targets (kept to match existing logic)
        self._q_des = None

        # ---------------------------------------------------------------------
        # Action buffers
        # ---------------------------------------------------------------------
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # ---------------------------------------------------------------------
        # Commands / targets
        # ---------------------------------------------------------------------
        # throwing_commands: [dist, theta_cos, phi] parameterization (see sampling functions)
        self.throwing_commands = torch.zeros(self.num_envs, 3, device=self.device)

        # target_positions is an *offset in env-local frame* (world computed via _target_world)
        self.target_positions = torch.zeros(self.num_envs, 3, device=self.device)

        # ---------------------------------------------------------------------
        # Per-episode bookkeeping buffers
        # ---------------------------------------------------------------------
        self.ball_distances = torch.zeros(self.num_envs, device=self.device) - 1.0
        self.not_released_ball = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        self.throwing_reward = torch.ones(self.num_envs, device=self.device) * -1
        self.throwing_reward_given = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.landing_time = torch.ones(self.num_envs, device=self.device) * -1
        self.target_hit_given = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.sum_open_hand_action = torch.zeros(self.num_envs, device=self.device)
        self.release_ball_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.release_target_dir = torch.zeros(self.num_envs, 3, device=self.device)
        self.released_ball_t = torch.zeros(self.num_envs, device=self.device) - 1.0

        # Stability penalties can become “sticky” within an episode
        self.stability_penalty_this_ep = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Curriculum + noise settings
        self.curriculum = True
        self.action_noise = 0.0
        self.prev_velocity = torch.zeros(self.num_envs, 3, device=self.device) - 1000

        # Optional safety termination flags
        self._joint_vel_violation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Optional ball trace debug visualization
        self.ball_trace_enabled = getattr(self.cfg, "ball_trace_enabled", False)
        self.ball_trace_env_ids = self._resolve_trace_env_ids(getattr(self.cfg, "ball_trace_env_ids", None))
        self.ball_trace_history_length = max(1, int(getattr(self.cfg, "ball_trace_history_length", 200)))
        self.ball_trace_color_override = getattr(self.cfg, "ball_trace_color", None)
        self.ball_trace_color = self._resolve_trace_color()
        self.ball_trace_thickness = float(getattr(self.cfg, "ball_trace_thickness", 4.0))
        self._ball_trace_points: dict[int, Deque[list[float]]] = {
            env_id: deque(maxlen=self.ball_trace_history_length)
            for env_id in self.ball_trace_env_ids
        } if self.ball_trace_env_ids is not None else {}
        self._ball_trace_draw_interface = None
        self._SimplexPointCls = None
        if self.ball_trace_enabled and self.sim.has_gui():
            from omni.debugdraw import acquire_debug_draw_interface
            from omni.debugdraw._debugDraw import SimplexPoint

            self._ball_trace_draw_interface = acquire_debug_draw_interface()
            self._SimplexPointCls = SimplexPoint
        self._ball_trace_default_color_int = self._rgba_to_argb(self.ball_trace_color)

        # ---------------------------------------------------------------------
        # Reward logging buffers (used for reporting + curriculum heuristics)
        # ---------------------------------------------------------------------
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "throwing",
                "projectile_rew",
                "stability",
                "action_rate_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "ballrel_rew",
            ]
        }
        self.reward_component_names = list(self._episode_sums.keys())
        self.reward_components = len(self.reward_component_names)
        self.reward_component_task_rew = ["throwing", "projectile_rew", "ballrel_rew"]

        # ---------------------------------------------------------------------
        # Config-dependent sampling ranges / task modes
        # ---------------------------------------------------------------------
        # arm_only: if True, disable “body” actions (treated downstream)
        self.body_actions = 0.0 if self.cfg.arm_only else 1.0

        # distance_throw: simplified task variant (fixed theta/phi; fixed distance range)
        if self.cfg.distance_throw:
            self.distance_range = [4, 4]
            self.theta_range = [1.0, 1.0]
        else:
            # Curriculum starts at an initial maximum distance and expands to max_throw_dist
            init_max_dist = getattr(self.cfg, "initial_max_throw_dist", self.cfg.max_throw_dist)
            init_max_dist = min(self.cfg.max_throw_dist, max(self.cfg.min_throw_dist, init_max_dist))
            self.distance_range = [self.cfg.min_throw_dist, init_max_dist]
            self.theta_range = [0.0, 1.0]

        # Curriculum initialization for target height range
        init_hmin, init_hmax = getattr(self.cfg, "initial_target_height_range", self.cfg.target_height_range)
        hmin = max(self.cfg.target_height_range[0], init_hmin)
        hmax = min(self.cfg.target_height_range[1], init_hmax)
        if hmax < hmin:
            hmax = hmin
        self.current_target_height_range = [hmin, hmax]

        # Precompute angular constants
        self.target_half_fov_rad = min(math.radians(self.cfg.target_fov_deg) / 2.0, math.pi)
        self.target_heading_offset_rad = math.radians(self.cfg.target_heading_offset_deg)
        self.robot_yaw_offset_rad = math.radians(self.cfg.robot_yaw_offset_deg)

        # If projectile motion is disabled, disable projected displacement observations as well
        if self.cfg.no_proj_motion:
            self.cfg.obs_estimdisplace = False

        # Initial command sampling across all envs
        self._sample_throwing_commands(torch.arange(self.num_envs, device=self.device))

        # Internal desired joint targets (for position-action semantics)
        # NOTE: `_robot` is set up in `_setup_scene` (called by base class), so we guard access.
        self._q_des = self._robot.data.joint_pos.clone() if hasattr(self, "_robot") else None

    # ---------------------------------------------------------------------
    # Scene
    # ---------------------------------------------------------------------
    def _setup_scene(self):
        """Create and register scene assets (robot, ball, target, sensors, terrain, lights).

        This is called by the IsaacLab environment lifecycle to build one template
        environment, then clone it `num_envs` times.
        """
        # Robot articulation
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Ball rigid object
        self.sphere_object = RigidObject(self.cfg.sphere_cfg)
        self.scene.rigid_objects["sphere"] = self.sphere_object

        # Target rigid object is only spawned for visualization (not required for physics)
        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.target_object = RigidObject(self.cfg.target_cfg)
            self.scene.rigid_objects["target"] = self.target_object

        # Contact sensor for collision detection (stability)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Terrain instantiation depends on scene cfg (num_envs/spacing)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone the source env many times for vectorized simulation
        self.scene.clone_environments(copy_from_source=False)

        # Optional: color-coding across envs (for visual debugging)
        self._apply_env_color_pairs()

        # Filter collisions against terrain prim to keep collision handling clean
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Dome light for stable HDRI lighting in rendering mode
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        light_cfg.func("/World/Light", light_cfg, orientation=(1.0, 0.0, 0.0, 0.0))

    # ---------------------------------------------------------------------
    # Joint index mapping
    # ---------------------------------------------------------------------
    def _build_alpha_joint_indices(self):
        """Build cached tensors mapping canonical joint names to DOF indices.

        Why this exists:
        - USD/URDF joint ordering can differ; policies assume a stable action layout.
        - We map by joint name once, then reuse the resulting index tensors.

        Populates:
            self._main_joint_ids: torso + both arms (15 DOFs expected)
            self._finger_joint_ids: finger joints (12 DOFs expected)
            self._right_arm_ids: right arm subset (for init randomization)
            self._other_main_joint_ids: main joints excluding right arm
            self._non_finger_joint_ids: alias for main joints
        """
        # If indices are already built, skip work.
        if (
            hasattr(self, "_main_joint_ids")
            and hasattr(self, "_finger_joint_ids")
            and hasattr(self, "_right_arm_ids")
            and hasattr(self, "_other_main_joint_ids")
        ):
            return

        # Gather joint names from the articulation and build reverse lookup.
        robot_joint_names = list(self._robot.data.joint_names)
        name_to_idx = {name: i for i, name in enumerate(robot_joint_names)}

        # Validate that the robot actually contains all expected canonical joints.
        missing = [j for j in CANONICAL_JOINTS if j not in name_to_idx]
        assert not missing, f"Missing joints in robot model: {missing}"

        # Convert canonical joint name ordering to index ordering.
        main_indices = [name_to_idx[j] for j in CANONICAL_MAIN_JOINTS]
        finger_indices = [name_to_idx[j] for j in CANONICAL_FINGER_JOINTS]

        right_arm_indices = [name_to_idx[j] for j in RIGHT_ARM_JOINTS]
        other_main_indices = [idx for idx in main_indices if idx not in right_arm_indices]

        # Store as torch tensors for efficient indexing in batched ops.
        self._main_joint_ids = torch.tensor(main_indices, device=self.device, dtype=torch.long)
        self._finger_joint_ids = torch.tensor(finger_indices, device=self.device, dtype=torch.long)
        self._right_arm_ids = torch.tensor(right_arm_indices, device=self.device, dtype=torch.long)
        self._other_main_joint_ids = torch.tensor(other_main_indices, device=self.device, dtype=torch.long)
        self._non_finger_joint_ids = self._main_joint_ids

        # Sanity checks (expected for this robot DOF layout)
        assert len(self._main_joint_ids) == 15
        assert len(self._finger_joint_ids) == 12
        assert len(self._right_arm_ids) == len(RIGHT_ARM_JOINTS)

    # ---------------------------------------------------------------------
    # Target WORLD position helper (single source of truth)
    # ---------------------------------------------------------------------
    def _target_world(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return target positions in world frame.

        `self.target_positions` is stored as an *offset in the env-local frame*.
        World position is obtained by adding per-env origin offsets from terrain.

        Args:
            env_ids: Optional subset of environment indices. If None, returns all envs.

        Returns:
            Tensor of shape (N, 3) with world-frame target positions.
        """
        if env_ids is None:
            return self.target_positions + self._terrain.env_origins
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        return self.target_positions[env_ids] + self._terrain.env_origins[env_ids]

    # ---------------------------------------------------------------------
    # Action processing (position-delta around default, not velocity integration)
    # ---------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process raw policy actions into joint position targets.

        Action semantics:
        - Main joints (15): position delta around `default_joint_pos`.
          `processed_q = q_default + action_scale * action`
        - Fingers (12): controlled by a single grip scalar (open vs closed posture)

        Notes:
        - We clip to joint limits if available from the articulation.
        - If rendering is enabled, the target marker is updated each step.
        """
        # Preserve action tensor for logging/penalties
        self.actions = actions.clone()
        self._actions = actions

        self._build_alpha_joint_indices()
        main_idx = self._main_joint_ids
        finger_idx = self._finger_joint_ids

        num_main = main_idx.shape[0]      # 15
        num_fingers = finger_idx.shape[0] # 12
        num_dofs = self._robot.data.joint_pos.shape[1]

        # Ensure the robot DOF layout matches the assumed "main + fingers" partition.
        assert num_main + num_fingers == num_dofs, (
            f"DOF mismatch: main={num_main}, fingers={num_fingers}, total={num_dofs}"
        )

        # Default joint pose (per-env) used as a stable reference
        q0 = self._robot.data.default_joint_pos

        # Start from current pose and overwrite the controlled DOFs
        self._processed_actions = self._robot.data.joint_pos.clone()

        # MAIN joints: interpret action as delta position (radians) around default.
        delta_q = self.cfg.action_scale * self._actions[:, :num_main]
        self._processed_actions[:, main_idx] = q0[:, main_idx] + delta_q

        # FINGERS: single scalar “grip” action controls open vs closed preset.
        grip_action = self._actions[:, num_main]  # (N,)
        closed = self._closed_finger_pose.unsqueeze(0).expand(self.num_envs, -1)
        open_ = torch.zeros_like(closed)

        # If grip_action <= 0 => closed hand; else open hand
        hand_close_mask = (grip_action <= 0).view(-1, 1)
        finger_targets = torch.where(hand_close_mask, closed, open_)
        self._processed_actions[:, finger_idx] = finger_targets

        # Clip joint targets to limits if the articulation provides them
        limits = getattr(self._robot.data, "joint_pos_limits", None)
        if limits is not None:
            lower, upper = limits[:, :, 0], limits[:, :, 1]
            self._processed_actions = torch.clamp(self._processed_actions, lower, upper)

        # Update visualization target marker if GUI rendering is active
        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(torch.arange(self.num_envs, device=self.device))

    def _apply_action(self) -> None:
        """Send processed joint position targets to the articulation controller."""
        self._robot.set_joint_position_target(self._processed_actions)

    # ---------------------------------------------------------------------
    # Observations
    # ---------------------------------------------------------------------
    def _get_observations(self) -> dict:
        """Assemble the policy observation vector.

        Observation is composed conditionally based on cfg toggles. Common pieces:
        - Base angular velocity, projected gravity, and optional roll
        - Throwing commands (dist, theta_cos, phi)
        - Joint position offsets (from default) + noise
        - Joint velocities + noise
        - Current actions
        - Ball state / projected displacement estimate (if enabled)
        - Additional task flags (ball released, time, etc.)

        Returns:
            Dict with key "policy" containing a (num_envs, obs_dim) tensor.
        """
        self._build_alpha_joint_indices()

        # Cache body ID sets for collision/stability computations.
        # Regex-based lookup happens once to avoid per-step overhead.
        if not hasattr(self, "non_feet_ids"):
            self.non_feet_ids, _ = self._contact_sensor.find_bodies("^(?!.*ankle).*$")
            self.hand_ids, _ = self._contact_sensor.find_bodies(".*(palm|five|six|three|four|zero|one|two).*")
            self.left_hand_ids, _ = self._contact_sensor.find_bodies(".*left_(palm).*")
            self.right_hand_ids, _ = self._contact_sensor.find_bodies(".*right_(palm).*")

        # Only use non-finger joints for proprioception in this design
        idx = self._non_finger_joint_ids

        # Proprioception: joint position offset from default + joint velocities
        joint_pos_info = self._robot.data.joint_pos[:, idx] - self._robot.data.default_joint_pos[:, idx]
        joint_vel_info = self._robot.data.joint_vel[:, idx]

        # Optional observation noise ranges (domain randomization)
        pos_noise_range = getattr(self.cfg, "joint_pos_noise_range", None)
        vel_noise_range = getattr(self.cfg, "joint_vel_noise_range", None)
        joint_pos_noise = (
            torch.zeros_like(joint_pos_info).uniform_(pos_noise_range[0], pos_noise_range[1])
            if pos_noise_range is not None
            else 0.0
        )
        joint_vel_noise = (
            torch.zeros_like(joint_vel_info).uniform_(vel_noise_range[0], vel_noise_range[1])
            if vel_noise_range is not None
            else 0.0
        )

        # Estimated projectile displacement to target (normalized) + time to closest approach
        estimated_displacement, estim_time = self.check_ball_displacement(
            torch.arange(self.num_envs, device=self.device)
        )
        estimated_displacement = 1 - estimated_displacement

        # Optional base roll computation from root quaternion (used only if cfg.obs_roll)
        w, x, y, z = self._robot.data.root_quat_w.T
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2)).to(self.device)

        # Inject noise into the displacement estimate as curriculum progresses
        noise_displace = (1 - self.action_noise) * estimated_displacement + self.action_noise * torch.randn_like(
            estimated_displacement
        )

        # Track previous base velocity (currently used as a buffer; preserved as-is)
        self.prev_velocity = self._robot.data.root_lin_vel_b.clone()

        # Ball linear velocity in world frame (vx, vy, vz) from body_state_w
        ball_data = self.sphere_object.data.body_state_w[:, 0, [7, 8, 9]]

        # Build the observation vector conditionally
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_ang_vel_b if self.cfg.obs_ang_vel else None,
                    self._robot.data.root_lin_vel_b if self.cfg.obs_lin_vel else None,
                    self._robot.data.projected_gravity_b if self.cfg.obs_proj_grav else None,
                    (roll.float() + (torch.rand_like(roll) * 0.02 - 0.01)).unsqueeze(-1) if self.cfg.obs_roll else None,
                    self.throwing_commands,
                    joint_pos_info + joint_pos_noise,
                    joint_vel_info + joint_vel_noise,
                    self._actions,
                    self.not_released_ball.unsqueeze(-1).float() if self.cfg.obs_notrelease else None,
                    noise_displace.unsqueeze(-1) if self.cfg.obs_estimdisplace else None,
                    estim_time.unsqueeze(-1) if self.cfg.obs_estimdisplace else None,
                    ball_data if self.cfg.obs_estimdisplace else None,
                )
                if tensor is not None
            ],
            dim=-1,
        )

        # Safety clip (prevents extreme values from destabilizing training)
        obs = torch.clip(obs, -1000, 1000)
        self._update_ball_trace()
        return {"policy": obs}

    # ---------------------------------------------------------------------
    # Rewards
    # ---------------------------------------------------------------------
    def _ensure_throw_hand_ids(self):
        """Resolve and cache body indices for the throwing hand and fingertips.

        This finds:
        - The palm body ID used to define "hand position" (release detection).
        - Fingertip IDs used for ball reset placement (grasp center approximation).

        It supports both left and right hand via `cfg.throw_hand_side`.
        """
        if hasattr(self, "_throw_hand_body_id") and hasattr(self, "_right_fingertip_ids"):
            return

        body_names = list(self._robot.data.body_names)
        hand_side = getattr(self.cfg, "throw_hand_side", "right").lower()
        preferred_palm = "right_palm" if hand_side == "right" else "left_palm"

        # Preferred: exact match for palm
        try:
            self._throw_hand_body_id = body_names.index(preferred_palm)
        except ValueError:
            # Fallback: any palm containing the hand_side keyword
            palm_candidates = [i for i, n in enumerate(body_names) if "palm" in n and hand_side in n]
            if len(palm_candidates) == 0:
                # Last resort: any palm at all
                palm_candidates = [i for i, n in enumerate(body_names) if "palm" in n]
            assert len(palm_candidates) > 0, "No palm body found"
            self._throw_hand_body_id = palm_candidates[-1]

        # Fingertip candidates used for computing an approximate grasp center
        if hand_side == "right":
            fingertip_ids, _ = self._contact_sensor.find_bodies(".*right_(thumb|index|pinky)_distal.*")
        else:
            fingertip_ids, _ = self._contact_sensor.find_bodies(".*left_(thumb|index|pinky)_distal.*")
        if len(fingertip_ids) == 0:
            fingertip_ids, _ = self._contact_sensor.find_bodies(".*(thumb|index|pinky).*distal.*")
        self._right_fingertip_ids = fingertip_ids

    def _get_rewards(self) -> torch.Tensor:
        """Compute per-step reward components.

        High-level structure:
        - Detect ball release event based on distance between hand and ball.
        - At release: compute projectile-to-target displacement estimate and log.
        - Compute stability reward/penalty via contact sensor + "ball not thrown" condition.
        - Add smoothness penalties (action rate, torques, joint accel).
        - Return a stacked reward vector ordered by `self.reward_component_names`.

        Returns:
            Tensor of shape (num_envs, num_reward_components).
        """
        self._ensure_throw_hand_ids()
        self._build_alpha_joint_indices()
        idx = self._non_finger_joint_ids

        # Ball and hand positions in world frame
        ball_pos = self.sphere_object.data.body_pos_w[:, 0, 0:3]
        hand_pos = self._robot.data.body_pos_w[:, self._throw_hand_body_id, :]
        dist_hand_ball = torch.norm(ball_pos - hand_pos, dim=1)

        # Release threshold (hand-ball distance beyond which we consider the ball released)
        release_threshold = float(getattr(self.cfg, "release_distance_threshold", 0.25))

        # ---------------------------------------------------------------------
        # Release event detection and projectile reward shaping (computed once per episode)
        # ---------------------------------------------------------------------
        release_mask = (dist_hand_ball > release_threshold) & (~self.throwing_reward_given)
        self.not_released_ball &= ~release_mask
        ball_released_envs = torch.nonzero(release_mask, as_tuple=False).flatten()

        # Initialize per-step reward buffers
        self.throwing_reward.zero_()
        projectile_rew = torch.zeros(self.num_envs, device=self.device)

        if ball_released_envs.numel() > 0:
            # Predicted minimum displacement to target and time of closest approach
            disp, landing_time = self.check_ball_displacement(ball_released_envs)

            # Convert normalized displacement into a reward in [0,1]
            proj_rew_vals = torch.clamp(1.0 - disp, min=0.0)

            # Assign reward only for envs that released the ball this step
            self.throwing_reward[ball_released_envs] = proj_rew_vals
            projectile_rew[ball_released_envs] = proj_rew_vals

            # Mark reward as “given” so it’s not recomputed every step
            self.throwing_reward_given[ball_released_envs] = True

            # Record predicted landing time in wall-clock episode time
            self.landing_time[ball_released_envs] = (
                landing_time + self.episode_length_buf[ball_released_envs] * self.step_dt
            )

            # Log release state for debugging/analysis
            self.release_ball_pos[ball_released_envs] = ball_pos[ball_released_envs]
            target_pos = self._target_world(ball_released_envs)
            self.release_target_dir[ball_released_envs] = target_pos - ball_pos[ball_released_envs]

        # ---------------------------------------------------------------------
        # Stability / collision handling
        # ---------------------------------------------------------------------
        ball_positions = self.sphere_object.data.root_pos_w.clone()
        ball_not_thrown_cond = (
            (torch.norm((hand_pos - ball_positions), dim=1) <= release_threshold) & (self.reset_buf == 1)
        )

        # `first_contact` indicates first-time contacts since last update
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)

        # Mask out hand collisions when the ball is still grasped (to avoid false positives)
        mask = torch.where(
            torch.norm((hand_pos - ball_positions), dim=1) <= release_threshold,
            torch.zeros(1, device=self.device),
            torch.ones(1, device=self.device),
        )
        if hasattr(self, "hand_ids"):
            first_contact[:, self.hand_ids] &= mask.view(-1, 1).expand(-1, len(self.hand_ids)).to(torch.bool)

        # Non-feet bodies used for “stability” contact checks (feet contacts are allowed)
        non_feet = getattr(self, "non_feet_ids", None)
        if non_feet is None:
            non_feet = torch.arange(first_contact.shape[1], device=self.device)

        collision = torch.sum(first_contact[:, non_feet], dim=1) > 0

        # Track whether an env has violated stability within the current episode
        self.stability_penalty_this_ep |= (ball_not_thrown_cond | collision)

        # Two stability reward modes: sparse vs non-sparse (controlled by cfg)
        if getattr(self.cfg, "nonsparse_stability_reward", False):
            stability_rew = ((~(ball_not_thrown_cond | collision)).float() / self.max_episode_length_s) * self.step_dt
        else:
            stability_rew = ((self.reset_buf == 1) & (~self.stability_penalty_this_ep)).float()

        # ---------------------------------------------------------------------
        # Smoothness penalties (action rate, torque, accel)
        # ---------------------------------------------------------------------
        action_rate = torch.mean(
            torch.square(self._actions[:, : idx.shape[0]] - self._previous_actions[:, : idx.shape[0]]), dim=1
        )
        joint_torques = torch.mean(torch.square(self._robot.data.applied_torque[:, idx]), dim=1)

        # Use sim-reported joint accelerations if available; else fallback to zeros
        joint_acc_data = getattr(self._robot.data, "joint_acc", None)
        if joint_acc_data is not None:
            joint_accel = torch.mean(torch.square(joint_acc_data[:, idx]), dim=1)
        else:
            joint_accel = torch.zeros_like(action_rate)

        # Optional: bonus reward at the moment the ball is released
        ballrel_rew = torch.zeros(self.num_envs, device=self.device)
        if ball_released_envs.numel() > 0:
            ballrel_rew[ball_released_envs] = getattr(self.cfg, "ball_release_reward_scale", 0.0)

        # ---------------------------------------------------------------------
        # Pack rewards (each scaled by cfg hyperparameters)
        # ---------------------------------------------------------------------
        rewards = {
            "throwing": torch.clamp(self.throwing_reward, min=0.0) * self.cfg.throwing_reward_scale,
            "projectile_rew": projectile_rew * float(getattr(self.cfg, "projectile_reward_scale", 0.0)),
            "stability": stability_rew * self.cfg.stability_reward_scale,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "ballrel_rew": ballrel_rew,
        }

        # ---------------------------------------------------------------------
        # Debug logs (percentages + release stats)
        # ---------------------------------------------------------------------
        if not hasattr(self, "extras"):
            self.extras = {}
        self.extras["log"] = self.extras.get("log", {})
        n_env = float(self.num_envs)
        self.extras["log"]["debug/collision_pct"] = 100.0 * collision.float().mean()
        self.extras["log"]["debug/ball_not_thrown_pct"] = 100.0 * ball_not_thrown_cond.float().mean()
        self.extras["log"]["debug/stability_penalty_pct"] = 100.0 * self.stability_penalty_this_ep.float().mean()

        # Per-step: only counts envs releasing this step (expected small for large num_envs)
        self.extras["log"]["debug/ball_released_pct"] = 100.0 * (ball_released_envs.numel() / max(n_env, 1.0))

        # Cumulative: fraction of envs that have released at least once this episode
        released_any = (~self.not_released_ball).float().mean() * 100.0
        self.extras["log"]["debug/ball_released_cum_pct"] = released_any

        # Curriculum update hook (kept “best-effort” with try/except as in original)
        try:
            self.update_curriculum(self.common_step_counter)
        except Exception:
            pass

        # Accumulate episode sums for reporting
        for k, v in rewards.items():
            self._episode_sums[k] += v

        # Update previous action buffer for action-rate penalty
        self._previous_actions = self._actions.clone()

        # Return reward vector in stable component order
        reward_vec = torch.stack([rewards[name] for name in self.reward_component_names], dim=1)
        return reward_vec

    # ---------------------------------------------------------------------
    # Ball trace helpers
    # ---------------------------------------------------------------------
    def _update_ball_trace(self) -> None:
        if not self.ball_trace_enabled or self.num_envs == 0:
            return
        ball_positions = self.sphere_object.data.root_pos_w[:, :3]
        for env_id in self.ball_trace_env_ids:
            pos = ball_positions[env_id].tolist()
            if env_id not in self._ball_trace_points:
                self._ball_trace_points[env_id] = deque(maxlen=self.ball_trace_history_length)
            self._ball_trace_points[env_id].append(pos)
        self._draw_ball_trace()

    def _draw_ball_trace(self) -> None:
        if self._ball_trace_draw_interface is None or self._SimplexPointCls is None:
            return
        clear_lines = getattr(self._ball_trace_draw_interface, "clear_lines", None)
        if callable(clear_lines):
            clear_lines()
        line_points = []
        for env_id, points in self._ball_trace_points.items():
            if len(points) < 2:
                continue
            point_list = list(points)
            starts = point_list[:-1]
            ends = point_list[1:]
            color_int = self._ball_trace_color_ints.get(env_id, self._ball_trace_default_color_int)
            for start, end in zip(starts, ends):
                start_pt = self._SimplexPointCls()
                start_pt.position = tuple(start)
                start_pt.color = color_int
                start_pt.width = self.ball_trace_thickness
                end_pt = self._SimplexPointCls()
                end_pt.position = tuple(end)
                end_pt.color = color_int
                end_pt.width = self.ball_trace_thickness
                line_points.extend([start_pt, end_pt])
        if line_points:
            self._ball_trace_draw_interface.draw_lines(line_points)

    @staticmethod
    def _rgba_to_argb(color: tuple[float, float, float, float]) -> int:
        r, g, b, a = color
        packed = 0
        for comp in (a, r, g, b):
            comp_int = max(0, min(255, int(round(comp * 255))))
            packed = (packed << 8) | comp_int
        return packed

    def _resolve_trace_env_ids(self, env_ids_cfg) -> list[int]:
        if env_ids_cfg is None:
            return list(range(self.num_envs))
        ids = []
        for entry in env_ids_cfg:
            try:
                idx = int(entry)
            except (TypeError, ValueError):
                continue
            ids.append(max(0, min(self.num_envs - 1, idx)))
        unique_ids = sorted(set(ids))
        return unique_ids if unique_ids else list(range(self.num_envs))

    def _resolve_trace_color(self, override=None) -> tuple[float, float, float, float]:
        color_source = override if override is not None else self.ball_trace_color_override
        if color_source is not None:
            color = tuple(color_source) if len(color_source) >= 3 else tuple(list(color_source) + [1.0])
        else:
            visual = getattr(self.cfg.sphere_cfg.spawn, "visual_material", None)
            color = getattr(visual, "diffuse_color", (0.0, 1.0, 0.0))
        if len(color) == 3:
            color = (color[0], color[1], color[2], 1.0)
        return tuple(color[:4])

    # ---------------------------------------------------------------------
    # Dones / Termination
    # ---------------------------------------------------------------------
    def _check_joint_velocity_violation(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check joint velocity safety constraint violations.

        If `cfg.joint_velocity_limit` is set, any env where any main joint exceeds
        that magnitude will be flagged as violating. This can be used either as:
        - termination condition (done), and/or
        - penalty reward (scale controlled by cfg)

        Returns:
            violation_mask: (num_envs,) bool tensor of cumulative violations within episode
            penalty: (num_envs,) float tensor for violation penalty
        """
        limit = getattr(self.cfg, "joint_velocity_limit", None)
        penalty_scale = getattr(self.cfg, "joint_velocity_penalty_scale", 0.0)
        if limit is None:
            violation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return violation, torch.zeros(self.num_envs, device=self.device)

        joint_vel = self._robot.data.joint_vel[:, self._main_joint_ids]
        violation = torch.any(torch.abs(joint_vel) > limit, dim=1)

        # Store cumulative episode violation (sticky)
        self._joint_vel_violation |= violation

        penalty = violation.float() * penalty_scale
        return self._joint_vel_violation.clone(), penalty

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and truncation signals.

        Termination conditions (terminated=True):
        - Ball hits ground after release
        - Ball travels too far from target (safety / runaway)
        - No-release timeout (agent never throws within fraction of episode)
        - Joint velocity violation (if enabled)

        Truncation condition (time_out=True):
        - Episode length reached

        Additionally logs a "success" boolean for evaluation:
        - success := hit_ground and distance(ball, target) < success_radius
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        ball_pos = self.sphere_object.data.root_pos_w  # (N, 3)
        target_pos = self._target_world()              # (N, 3)

        released = ~self.not_released_ball
        hit_ground = (ball_pos[:, 2] < 0.05) & released

        # Too-far is a heuristic termination to prevent runaway episodes
        too_far = released & (torch.norm(ball_pos[:, :2] - target_pos[:, :2], dim=1) > self.cfg.max_throw_dist + 1.0)

        joint_vel_violation, _ = self._check_joint_velocity_violation()

        # If ball never releases, terminate after a portion of the episode
        timeout_frac = getattr(self.cfg, "no_release_timeout_frac", 0.75)
        no_release_timeout = (self.episode_length_buf > timeout_frac * self.max_episode_length) & (~released)

        terminated = hit_ground | too_far | no_release_timeout | joint_vel_violation

        # Success definition: landed within a radius of the target
        success_radius = float(getattr(self.cfg, "success_radius", 0.3))
        success = hit_ground & (torch.norm(ball_pos - target_pos, dim=1) < success_radius)

        # Store termination info for logging / debugging
        self.extras["termination"] = {
            "terminated": terminated,
            "truncated": time_out,
            "success": success,
            "joint_vel_violation": joint_vel_violation,
        }
        return terminated, time_out

    # ---------------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset a subset (or all) environments to start a new episode.

        Responsibilities:
        - Reset robot + base class env state
        - Reset bookkeeping buffers (release flags, reward flags, logs)
        - Sample new throwing commands and compute corresponding target offsets
        - Randomize joint initial positions/velocities (cfg-controlled)
        - Reset robot root pose and joint states in simulator
        - Reset ball to the throwing hand (closed fingers)
        - Update rendered target marker if rendering enabled
        """
        # Treat None or full-length list as "reset all"
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Reset robot internal buffers and base env bookkeeping
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.ball_trace_enabled:
            reset_env_ids = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)
            for env_id in reset_env_ids:
                if env_id in self._ball_trace_points:
                    self._ball_trace_points[env_id].clear()
            if self._ball_trace_draw_interface is not None:
                clear_lines = getattr(self._ball_trace_draw_interface, "clear_lines", None)
                if callable(clear_lines):
                    clear_lines()

        self._build_alpha_joint_indices()
        grip_action_index = self._main_joint_ids.shape[0]  # finger/grip scalar action index

        # Initialize actions (grip starts closed by setting grip action to -1)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._actions[env_ids, grip_action_index] = -1.0
        self._previous_actions[env_ids, grip_action_index] = -1.0

        # Clear episode logging sums
        for k in self._episode_sums.keys():
            self._episode_sums[k][env_ids] = 0.0

        # Reset per-episode flags and buffers
        self.stability_penalty_this_ep[env_ids] = False
        self.ball_distances[env_ids] = -1.0
        self.release_ball_pos[env_ids] = 0.0
        self.release_target_dir[env_ids] = 0.0
        self.throwing_reward[env_ids] = -1.0
        self.landing_time[env_ids] = -1.0
        self.throwing_reward_given[env_ids] = False
        self.target_hit_given[env_ids] = False
        self.not_released_ball[env_ids] = True
        self.sum_open_hand_action[env_ids] = 0.0
        self.released_ball_t[env_ids] = -1.0
        self._joint_vel_violation[env_ids] = False

        # Sample commands + compute target offsets
        self._sample_throwing_commands(env_ids)
        self.target_positions[env_ids] = self.calculate_target_offset(env_ids)

        # Joint reset: start at default pose + optional random noise
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self._robot.data.default_joint_vel[env_ids])

        # Fingers start closed to hold the ball at reset
        joint_pos[:, self._finger_joint_ids] = self._closed_finger_pose

        # Randomized initialization: right arm joints
        right_arm_range = float(getattr(self.cfg, "right_arm_init_range", 0.0))
        if right_arm_range > 0.0 and self._right_arm_ids.numel() > 0:
            right_arm_noise = torch.zeros_like(joint_pos[:, self._right_arm_ids])
            right_arm_noise.uniform_(-right_arm_range, right_arm_range)
            joint_pos[:, self._right_arm_ids] += right_arm_noise

        # Randomized initialization: other main joints + fingers
        other_joint_range = float(getattr(self.cfg, "other_joint_init_range", 0.0))
        other_joint_ids = torch.cat((self._other_main_joint_ids, self._finger_joint_ids))
        if other_joint_range > 0.0 and other_joint_ids.numel() > 0:
            other_noise = torch.zeros_like(joint_pos[:, other_joint_ids])
            other_noise.uniform_(-other_joint_range, other_joint_range)
            joint_pos[:, other_joint_ids] += other_noise

        # Global joint position noise
        pos_noise_range = getattr(self.cfg, "joint_pos_noise_range", None)
        if pos_noise_range is not None:
            joint_pos += torch.zeros_like(joint_pos).uniform_(pos_noise_range[0], pos_noise_range[1])

        # Global joint velocity noise
        vel_noise_range = getattr(self.cfg, "joint_vel_noise_range", None)
        if vel_noise_range is not None:
            joint_vel += torch.zeros_like(joint_vel).uniform_(vel_noise_range[0], vel_noise_range[1])

        # Clamp joints to limits if provided by sim
        limits = getattr(self._robot.data, "joint_pos_limits", None)
        if limits is not None:
            lower = limits[env_ids, :, 0]
            upper = limits[env_ids, :, 1]
            joint_pos = torch.max(torch.min(joint_pos, upper), lower)

        # Root reset: start from default and zero out linear/angular velocity
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 7:] = 0.0

        # Optional yaw offset applied to root orientation (quat multiplication)
        if self.robot_yaw_offset_rad != 0.0:
            offset_w = math.cos(self.robot_yaw_offset_rad * 0.5)
            offset_z = math.sin(self.robot_yaw_offset_rad * 0.5)
            offset_quat = torch.zeros_like(default_root_state[:, 3:7])
            offset_quat[:, 0] = offset_w
            offset_quat[:, 3] = offset_z
            default_root_state[:, 3:7] = self._quat_multiply(offset_quat, default_root_state[:, 3:7])

        # Place robot in correct env origin
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # Write root + joint state into simulator
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Sync desired joint targets
        self._q_des = self._robot.data.joint_pos.clone()

        # Reset ball into the throwing hand
        self._reset_ball_to_hand(env_ids)

        # Render target marker if GUI rendering is active
        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(env_ids)

    def _reset_ball_to_hand(self, env_ids: Sequence[int] | torch.Tensor):
        """Reset the ball pose near the throwing hand (grasp pose).

        Strategy:
        - Start from the ball's default root state.
        - Place it at a fixed offset from palm (in palm frame), transformed to world.
        - If fingertip IDs are available, refine the ball offset using the
          fingertips' mean position to estimate a grasp center.

        Args:
            env_ids: Subset of environment indices to reset.
        """
        if env_ids is None or len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._ensure_throw_hand_ids()

        ball_state = self.sphere_object.data.default_root_state.clone()[env_ids]
        hand_pos = self._robot.data.body_pos_w[env_ids, self._throw_hand_body_id]
        hand_quat = self._robot.data.body_quat_w[env_ids, self._throw_hand_body_id]

        # Base offset from palm to place ball (tuned for this robot hand geometry)
        forward_offset_local = torch.tensor([0.0, 0.08, 0.04], device=self.device)
        forward_offset_world = self._quat_apply(hand_quat, forward_offset_local.expand(len(env_ids), -1))
        ball_offset = forward_offset_world

        # Optional refinement using fingertips: pull ball toward centroid direction
        fingertip_ids = getattr(self, "_right_fingertip_ids", [])
        if len(fingertip_ids) > 0:
            fingertip_ids = torch.as_tensor(fingertip_ids, device=self.device, dtype=torch.long)
            fingertip_positions = self._robot.data.body_pos_w[env_ids][:, fingertip_ids, :]
            grasp_center = fingertip_positions.mean(dim=1)

            offset = grasp_center - hand_pos
            dist = torch.norm(offset, dim=1, keepdim=True)
            mask = dist.squeeze(-1) > 1e-4
            if mask.any():
                dir_norm = offset[mask] / dist[mask]
                ball_offset[mask] = dir_norm * 0.05

        # Apply the final ball pose and clear velocity
        ball_pos = hand_pos + ball_offset
        ball_state[:, :3] = ball_pos
        ball_state[:, 3:7] = self.sphere_object.data.default_root_state[env_ids, 3:7]
        ball_state[:, 7:] = 0.0
        self.sphere_object.write_root_state_to_sim(ball_state, env_ids)

    # ---------------------------------------------------------------------
    # Curriculum (unchanged)
    # ---------------------------------------------------------------------
    def update_curriculum(self, iter):
        """Heuristic curriculum update for sampling ranges and observation noise.

        Uses logged episode reward components (stability + throwing) to decide
        when to expand target distance and height ranges. Also increases
        `action_noise` gradually over time.

        Args:
            iter: Global iteration / step counter (passed from env loop).
        """
        log_dict = self.extras.get("log", {}) if hasattr(self, "extras") else {}
        stability_value = log_dict.get("Episode_Reward/stability", None)
        throwing_value = log_dict.get("Episode_Reward/throwing", None)
        if stability_value is None or throwing_value is None:
            return

        # Progressively increase noise on displacement estimates
        self.action_noise = min(1.0, self.action_noise + 0.001)

        # If both task reward and stability are above thresholds, expand curriculum
        if (throwing_value > self.cfg.r_throw_thresh and stability_value > self.cfg.r_stability_thresh) or (iter >= 750):
            dist_step = getattr(self.cfg, "curriculum_distance_increment", 0.01)
            height_step = getattr(self.cfg, "curriculum_height_increment", 0.01)

            if self.cfg.distance_throw:
                self.distance_range = [
                    min(self.cfg.max_throw_dist, self.distance_range[0] + dist_step),
                    min(self.cfg.max_throw_dist, self.distance_range[1] + dist_step),
                ]
            else:
                self.distance_range = [
                    self.distance_range[0],
                    min(self.cfg.max_throw_dist, self.distance_range[1] + dist_step),
                ]
                # Expand allowable theta_cos upper range
                self.theta_range = [min(0.0, self.theta_range[0]), min(1.0, self.theta_range[1] + 0.01)]

            # Expand target height upper bound
            new_height_max = min(self.cfg.target_height_range[1], self.current_target_height_range[1] + height_step)
            self.current_target_height_range[1] = new_height_max

    # ---------------------------------------------------------------------
    # Command sampling + target offset
    # ---------------------------------------------------------------------
    def _sample_throwing_commands(self, env_ids: torch.Tensor | None):
        """Sample (dist, theta_cos, phi) throwing commands for specified envs.

        Parameterization:
        - dist: radial distance to target
        - theta_cos: cos(theta), where theta is polar angle from +Z axis
        - phi: azimuth in the robot/body frame, limited by FOV and hand side

        Hand-side behavior:
        - Right-hand throws sample phi in [0, +FOV/2]
        - Left-hand throws sample phi in [-FOV/2, 0]

        Supports fixed distance/height overrides if cfg provides them.

        Args:
            env_ids: Subset of env indices. If None, samples for all envs.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        command_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Azimuth sampling within target FOV
        if self.target_half_fov_rad <= 0.0:
            phi_samples = torch.zeros_like(self.throwing_commands[command_ids, 2])
        else:
            hand_side = getattr(self.cfg, "throw_hand_side", "right").lower()
            if hand_side == "right":
                phi_low, phi_high = 0.0, self.target_half_fov_rad
            else:
                phi_low, phi_high = -self.target_half_fov_rad, 0.0
            phi_samples = torch.zeros_like(self.throwing_commands[command_ids, 2]).uniform_(phi_low, phi_high)

        fixed_dist = getattr(self.cfg, "fixed_throw_dist", None)
        fixed_height = getattr(self.cfg, "fixed_target_height", None)
        sequential_mode = getattr(self.cfg, "sequential_distance_mode", False) and fixed_dist is None

        # Distance sampling (curriculum-controlled unless fixed)
        if sequential_mode:
            start = float(getattr(self.cfg, "sequential_distance_start", self.cfg.min_throw_dist))
            step = float(getattr(self.cfg, "sequential_distance_step", 0.1))
            step = max(step, 0.0)
            dist_final = start + command_ids.to(torch.float32) * step
            dist_final = torch.clamp(dist_final, min=self.cfg.min_throw_dist, max=self.cfg.max_throw_dist)
        elif fixed_dist is not None:
            dist_final = torch.full_like(self.throwing_commands[command_ids, 0], float(fixed_dist))
        else:
            dist_min, dist_max = self.distance_range
            dist_final = torch.zeros_like(self.throwing_commands[command_ids, 0]).uniform_(dist_min, dist_max)

        # Height sampling (curriculum-controlled unless fixed)
        if fixed_height is not None:
            z_clamped = torch.full_like(self.throwing_commands[command_ids, 0], float(fixed_height))
        else:
            z_min, z_max = self.current_target_height_range
            z_samples = torch.zeros_like(self.throwing_commands[command_ids, 0]).uniform_(z_min, z_max)
            # Clamp height to remain feasible given distance (avoid invalid theta)
            z_clamped = torch.clamp(z_samples, min=0.0, max=dist_final.max() - 0.05)

        # Convert height into theta_cos = z / dist
        theta_cos = torch.clamp(z_clamped / torch.clamp(dist_final, min=1e-3), -1.0, 1.0)

        # Distance throw variant pins theta and phi
        if self.cfg.distance_throw:
            theta_cos = torch.ones_like(theta_cos)
            phi_samples = torch.zeros_like(phi_samples)

        # Write commands into buffer
        self.throwing_commands[command_ids, 0] = dist_final
        self.throwing_commands[command_ids, 1] = theta_cos
        self.throwing_commands[command_ids, 2] = phi_samples

    def calculate_target_offset(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Convert throwing commands into an env-local target offset vector.

        Steps:
        1) Interpret (dist, theta_cos, phi) as spherical coordinates in body frame.
        2) Convert to Cartesian in local frame.
        3) Rotate by base yaw + configured heading offset to get world-aligned offset.
        4) Return env-local offset that will later be converted to world via `_target_world`.

        Args:
            env_ids: Environment indices for which to compute target offsets.

        Returns:
            Tensor (len(env_ids), 3): target offset vectors in env-local frame.
        """
        command_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        dist = self.throwing_commands[command_ids, 0]
        theta_cos = torch.clamp(self.throwing_commands[command_ids, 1], -1.0, 1.0)
        phi_body = self.throwing_commands[command_ids, 2]
        theta = torch.arccos(theta_cos)

        # Local (body-frame) Cartesian target direction
        x_local = dist * torch.sin(theta) * torch.cos(phi_body)
        y_local = dist * torch.sin(theta) * torch.sin(phi_body)
        z_local = dist * torch.cos(theta)

        # Rotate by base yaw and heading offset to align with world axes
        base_yaw = self._get_base_yaw(command_ids)
        yaw = base_yaw + self.target_heading_offset_rad
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

        x_world = x_local * cos_yaw - y_local * sin_yaw
        y_world = x_local * sin_yaw + y_local * cos_yaw
        return torch.stack((x_world, y_world, z_local), dim=1)

    def check_ball_displacement(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate normalized closest-approach displacement from projectile trajectory to target.

        This predicts the ball's ballistic trajectory (no air resistance),
        evaluates distance to target across sampled time steps, and returns:
        - the minimum normalized displacement (distance / throw_distance)
        - the time at which that minimum occurs

        Args:
            env_ids: Environment indices (subset) for which to compute displacement.

        Returns:
            disp_min: (len(env_ids),) minimum normalized displacement in [0, 1]
            time_at_min: (len(env_ids),) time (seconds) at which min displacement occurs
        """
        ball_data = self.sphere_object.data.body_state_w[env_ids, 0, :10]
        target_positions = self._target_world(env_ids)  # Single source of truth for target world position

        if self.cfg.air_resistance:
            raise NotImplementedError
        else:
            # Extract initial velocities and height
            vx0, vy0, vz0 = ball_data[:, 7], ball_data[:, 8], ball_data[:, 9]
            a = 9.81  # gravity magnitude

            z0 = ball_data[:, 2]
            z0_minus_0_03 = torch.clamp(z0 - 0.03, min=0.0)
            sqrt_term = torch.sqrt(vz0**2 + 2 * a * z0_minus_0_03)

            # Solve time-to-ground roots (quadratic) and take the valid max
            t1 = (-vz0 + sqrt_term) / -a
            t2 = (-vz0 - sqrt_term) / -a
            tz = torch.max(t1, t2)
            tz = torch.clamp(tz, min=0.0)

            # Sample along trajectory from t=0 to t=tz
            steps = 100
            frac = torch.linspace(0, 1, steps=steps, device=self.device).unsqueeze(1)
            time_tensor = frac * tz.unsqueeze(0)

            # Ballistic equations
            new_x = ball_data[:, 0].unsqueeze(0) + vx0.unsqueeze(0) * time_tensor
            new_y = ball_data[:, 1].unsqueeze(0) + vy0.unsqueeze(0) * time_tensor
            new_z = (
                ball_data[:, 2].unsqueeze(0)
                + vz0.unsqueeze(0) * time_tensor
                - 0.5 * a * time_tensor**2
            )

            # Normalize distance error by commanded throw distance (scale-invariant reward)
            distance_command = self.throwing_commands[env_ids, 0].unsqueeze(0)
            disp_matrix = torch.sqrt(
                (new_x - target_positions[:, 0].unsqueeze(0)) ** 2
                + (new_y - target_positions[:, 1].unsqueeze(0)) ** 2
                + (new_z - target_positions[:, 2].unsqueeze(0)) ** 2
            ) / torch.clamp(distance_command, min=1e-3)

            disp_matrix = torch.clamp(disp_matrix, max=1.0)
            disp_min = torch.amin(disp_matrix, dim=0)
            idx_min = torch.argmin(disp_matrix, dim=0)

            batch_idx = torch.arange(len(env_ids), device=self.device)
            time_at_min = time_tensor[idx_min, batch_idx]

        return disp_min, time_at_min

    # ---------------------------------------------------------------------
    # Rendering target
    # ---------------------------------------------------------------------
    def initialise_target_for_rendering(self, env_ids):
        """Update the rendered target object pose for the specified envs.

        This is visualization-only: the target object is kinematic and exists
        primarily as a UI/GUI marker for debugging.

        Args:
            env_ids: Environment indices to update.
        """
        target_default_state = self.target_object.data.default_root_state.clone()[env_ids]
        target_default_state[:, 7:] = 0.0

        # Place target at consistent world location
        target_default_state[:, 0:3] = self._target_world(env_ids)

        # Orientation tweak: rotate 90 degrees about Z so board normal points +Y
        half_angle = math.pi / 4.0
        target_default_state[:, 3:7] = torch.tensor(
            [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)],
            device=self.device,
        )

        self.target_object.write_root_state_to_sim(target_default_state, env_ids)

    # ---------------------------------------------------------------------
    # Misc: yaw / quats / colors
    # ---------------------------------------------------------------------
    def _get_base_yaw(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Extract base yaw (heading) from the robot root quaternion.

        Args:
            env_ids: Environment indices.

        Returns:
            Tensor (len(env_ids),) yaw angles in radians.
        """
        command_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        base_quats = self._robot.data.root_quat_w[command_ids]
        w, x, y, z = base_quats.unbind(dim=1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.where(torch.isnan(yaw), torch.zeros_like(yaw), yaw)

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiply (Hamilton product) for batches of quaternions.

        Args:
            q1: (N, 4) quaternion tensor [w, x, y, z]
            q2: (N, 4) quaternion tensor [w, x, y, z]

        Returns:
            (N, 4) product quaternion q = q1 ⊗ q2
        """
        w1, x1, y1, z1 = q1.unbind(dim=1)
        w2, x2, y2, z2 = q2.unbind(dim=1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack((w, x, y, z), dim=1)

    @staticmethod
    def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Apply quaternion rotation to vectors.

        Args:
            quat: (N, 4) quaternions [w, x, y, z]
            vec: (N, 3) vectors

        Returns:
            (N, 3) rotated vectors.
        """
        w, x, y, z = quat.unbind(dim=1)
        vx, vy, vz = vec.unbind(dim=1)
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rx = (ww + xx - yy - zz) * vx + 2 * ((xy - wz) * vy + (xz + wy) * vz)
        ry = 2 * ((xy + wz) * vx + (ww - xx + yy - zz) * vy + (yz - wx) * vz)
        rz = 2 * ((xz - wy) * vx + (yz + wx) * vy + (ww - xx - yy + zz) * vz)
        return torch.stack((rx, ry, rz), dim=1)

    def _apply_env_color_pairs(self):
        """Assign per-environment colors to sphere and target for visual debugging.

        This iterates over cloned env prim paths and sets diffuse colors on the
        corresponding material shaders.
        """
        env_colors = self._generate_env_color_palette(self.scene.cfg.num_envs)
        if len(env_colors) == 0:
            return

        stage = stage_utils.get_current_stage()
        sphere_material_path = getattr(self.cfg.sphere_cfg.spawn, "visual_material_path", "material") or "material"

        target_material_path = None
        target_cfg = getattr(self.cfg, "target_cfg", None)
        target_available = hasattr(self, "target_object") and target_cfg is not None and target_cfg.spawn is not None
        if target_available:
            target_material_path = getattr(target_cfg.spawn, "visual_material_path", "material") or "material"

        # Choose an index ordering that spreads colors across env IDs
        color_indices = self._color_distribution_indices(len(env_colors))

        for env_index, env_path in enumerate(self.scene.env_prim_paths):
            if env_index >= len(env_colors):
                break
            color = env_colors[color_indices[env_index]]

            # Sphere shader path depends on USD structure produced by IsaacLab
            sphere_shader_path = f"{env_path}/sphere/geometry/{sphere_material_path}/Shader"
            self._set_shader_color(stage, sphere_shader_path, color)

            # Target exists only in rendering mode
            if target_material_path is not None:
                target_shader_path = f"{env_path}/target/geometry/{target_material_path}/Shader"
                self._set_shader_color(stage, target_shader_path, color)
            color_with_alpha = (color[0], color[1], color[2], 1.0)
            self._ball_trace_color_ints[env_index] = self._rgba_to_argb(color_with_alpha)

    def _generate_env_color_palette(self, num_envs: int) -> list[tuple[float, float, float]]:
        """Generate a visually distinct color palette for N environments.

        Uses HSV hue sweep with fixed saturation/value.
        """
        if num_envs <= 0:
            return []
        saturation, value = 0.75, 0.9
        colors: list[tuple[float, float, float]] = []
        for env_index in range(num_envs):
            hue = (env_index / max(num_envs, 1)) % 1.0
            colors.append(self._hsv_to_rgb(hue, saturation, value))
        return colors

    def _initialize_ball_trace_color_ints(self, num_envs: int) -> dict[int, int]:
        """Pre-fill RGB ints for each env's color palette (matches `_apply_env_color_pairs`)."""
        palette = self._generate_env_color_palette(num_envs)
        color_ints = {}
        for idx, color in enumerate(palette):
            color_with_alpha = (color[0], color[1], color[2], 1.0)
            color_ints[idx] = self._rgba_to_argb(color_with_alpha)
        return color_ints

    @staticmethod
    def _set_shader_color(stage, shader_path: str, color: tuple[float, float, float]):
        """Set diffuse color on a USD shader prim (if it exists)."""
        prim = stage.GetPrimAtPath(shader_path)
        if prim.IsValid():
            sim_utils.safe_set_attribute_on_usd_prim(prim, "inputs:diffuseColor", color, camel_case=False)

    @staticmethod
    def _hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[float, float, float]:
        """Convert HSV color to RGB (utility for palette generation)."""
        hue = hue % 1.0
        i = int(hue * 6.0)
        f = hue * 6.0 - i
        i = i % 6
        p = value * (1.0 - saturation)
        q = value * (1.0 - f * saturation)
        t = value * (1.0 - (1.0 - f) * saturation)

        if i == 0:
            r, g, b = value, t, p
        elif i == 1:
            r, g, b = q, value, p
        elif i == 2:
            r, g, b = p, value, t
        elif i == 3:
            r, g, b = p, q, value
        elif i == 4:
            r, g, b = t, p, value
        else:
            r, g, b = value, p, q
        return (r, g, b)

    def _color_distribution_indices(self, num_envs: int) -> list[int]:
        """Compute an index permutation that spreads colors across env indices.

        Uses a step size that is coprime with `num_envs` so the modular sequence
        visits each index exactly once (full cycle).
        """
        if num_envs <= 1:
            return list(range(num_envs))

        # Start with a coarse step; adjust until it's coprime with num_envs
        step = max(1, num_envs // 3 or 1)
        while gcd(step, num_envs) != 1:
            step += 1

        order = []
        idx = 0
        for _ in range(num_envs):
            order.append(idx)
            idx = (idx + step) % num_envs
        return order
