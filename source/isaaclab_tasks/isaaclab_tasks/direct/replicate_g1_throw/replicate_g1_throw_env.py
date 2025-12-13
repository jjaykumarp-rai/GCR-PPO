# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .replicate_g1_throw_env_cfg import ReplicateG1ThrowEnvCfg

###### part of CDR-1630 ######
import gymnasium as gym
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaacsim.core.utils.stage as stage_utils
from math import gcd
import torch.nn.functional as F
from .alpha_utils import (
    TORSO_JOINTS,
    RIGHT_ARM_JOINTS,
    LEFT_ARM_JOINTS,
    FINGER_JOINTS,
)

CANONICAL_MAIN_JOINTS = TORSO_JOINTS + RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS
CANONICAL_FINGER_JOINTS = FINGER_JOINTS
CANONICAL_JOINTS = CANONICAL_MAIN_JOINTS + CANONICAL_FINGER_JOINTS

MAIN_JOINT_INDICES = torch.tensor(
    [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13],
    dtype=torch.long,
)

FINGER_JOINT_INDICES = torch.tensor(
    [20, 18, 19, 26, 24, 25, 17, 15, 16, 23, 21, 22],
    dtype=torch.long,
)

CLOSED_FINGER_TARGETS = (
    0.85,
    0.85,
    0.85,
    0.65,
    0.65,
    0.65,
    0.85,
    0.85,
    0.85,
    0.65,
    0.65,
    0.65,
)

class ReplicateG1ThrowEnv(DirectRLEnv):
    cfg: ReplicateG1ThrowEnvCfg

    def __init__(self, cfg: ReplicateG1ThrowEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # set joints in canonical order
        self._build_alpha_joint_indices()
        # Always use the resolved joint indices from the robot, not the hard-coded fallbacks.
        self.main_joint_indices = self._main_joint_ids
        self.finger_joint_indices = self._finger_joint_ids
        self._closed_finger_pose = torch.tensor(CLOSED_FINGER_TARGETS, device=self.device)

        self.curriculum = True

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        self.throwing_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_positions = torch.zeros(self.num_envs, 3, device=self.device)

        self.ball_distances = torch.zeros((self.num_envs), device=self.device) -1 

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "throwing",
                "stability",
                "action_rate_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "ballrel_rew",
                "vel_align_rew",
            ]
        }

        self.reward_components = len(self._episode_sums.keys())
        self.reward_component_names = list(self._episode_sums.keys())
        self.reward_component_task_rew = ["throwing", "ballrel_rew", "vel_align_rew"]

        # if self.cfg.baseh_rew:
        #     self.reward_component_names += ["baseh_rew"]
        #     self.reward_components += 1
        #     self.reward_component_task_rew += ["baseh_rew"]
        #     self._episode_sums["baseh_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # if self.cfg.energy_rew:
        #     self.reward_component_names += ["energy_rew"]
        #     self.reward_components += 1
        #     self.reward_component_task_rew += ["energy_rew"]
        #     self._episode_sums["energy_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # if self.cfg.ballrel_rew:
        #     self.reward_component_names += ["ballrel_rew"]
        #     self.reward_components += 1
        #     self.reward_component_task_rew += ["ballrel_rew"]
        #     self._episode_sums["ballrel_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # if self.cfg.bodymo_rew:
        #     self.reward_component_names += ["bodymo_rew"]
        #     self.reward_components += 1
        #     self.reward_component_task_rew += ["bodymo_rew"]
        #     self._episode_sums["bodymo_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # if self.cfg.lftarm_rew:
        #     self.reward_component_names += ["lftarm_rew"]
        #     self.reward_components += 1
        #     self.reward_component_task_rew += ["lftarm_rew"]
        #     self._episode_sums["lftarm_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # if self.cfg.rgtarmrel_rew:
        #     self.reward_component_names += ["rgtarmrel_rew"]
        #     self.reward_components += 1
        #     self.reward_component_task_rew += ["rgtarmrel_rew"]
        #     self._episode_sums["rgtarmrel_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.not_released_ball = torch.ones((self.num_envs), device=self.device).bool()
        self.throwing_reward = torch.ones((self.num_envs), device=self.device)*-1 # by default it is -1 if not thrown
        self.landing_time = torch.ones((self.num_envs), device=self.device)*-1
        self.throwing_reward_given = torch.zeros((self.num_envs), device=self.device).bool()
        self.target_hit_given = torch.zeros((self.num_envs), device=self.device).bool()
        self.sum_open_hand_action = torch.zeros((self.num_envs), device=self.device)
        self.release_ball_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.release_target_dir = torch.zeros((self.num_envs, 3), device=self.device)
        
        # removed stability related things
        # self.min_base_height = 0.38 
        self.stability_penalty_this_ep = torch.zeros((self.num_envs), device=self.device).bool()

        if self.cfg.arm_only:
            self.body_actions = 0.
        else:
            self.body_actions = 1. # max is 100%

        # removed stability related things
        # self.stability_timer = 2. # max is 2 (seconds)

        if self.cfg.distance_throw:
            self.distance_range = [4, 4]
            self.theta_range = [1.0, 1.0]
        else:
            init_max_dist = getattr(self.cfg, "initial_max_throw_dist", self.cfg.max_throw_dist)
            init_max_dist = min(self.cfg.max_throw_dist, max(self.cfg.min_throw_dist, init_max_dist))
            self.distance_range = [self.cfg.min_throw_dist, init_max_dist]
            self.theta_range = [0., 1.]
        init_height_min, init_height_max = getattr(
            self.cfg, "initial_target_height_range", self.cfg.target_height_range
        )
        height_min = max(self.cfg.target_height_range[0], init_height_min)
        height_max = min(self.cfg.target_height_range[1], init_height_max)
        if height_max < height_min:
            height_max = height_min
        self.current_target_height_range = [height_min, height_max]

        self.target_half_fov_rad = min(math.radians(self.cfg.target_fov_deg) / 2.0, math.pi)
        self.target_heading_offset_rad = math.radians(self.cfg.target_heading_offset_deg)
        # Use configured robot yaw offset; do not rotate the scene.
        self.robot_yaw_offset_rad = math.radians(self.cfg.robot_yaw_offset_deg)
        self._sample_throwing_commands(torch.arange(self.num_envs, device=self.device))
        self.released_ball_t = torch.zeros((self.num_envs), device=self.device)-1
        self.action_noise = 0.

        if self.cfg.no_proj_motion:
            self.cfg.obs_estimdisplace = False

        self.prev_velocity = torch.zeros((self.num_envs, 3), device=self.device)-1000
        
        self.default_root_states = torch.zeros((self.num_envs, 3), device=self.device)
        self._joint_vel_violation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _build_alpha_joint_indices(self):
        """Build canonical joint index maps for Alpha (torso, arms, fingers)."""

        # Already built? Skip.
        if hasattr(self, "_main_joint_ids") and hasattr(self, "_finger_joint_ids"):
            return

        # Robot’s raw joint order from Isaac
        robot_joint_names = list(self._robot.data.joint_names)
        name_to_idx = {name: i for i, name in enumerate(robot_joint_names)}

        # Basic sanity: check all required joints are present
        missing = [j for j in CANONICAL_JOINTS if j not in name_to_idx]
        assert not missing, f"Missing joints in robot model: {missing}"

        # Indices in the *robot* DOF array, ordered canonically
        main_indices = [name_to_idx[j] for j in CANONICAL_MAIN_JOINTS]      # 15: TJ1 + RJ1..RJ7 + LJ1..LJ7
        finger_indices = [name_to_idx[j] for j in CANONICAL_FINGER_JOINTS]  # 12: RTJ*/RIJ*/RPJ*/LTJ*/LIJ*/LPJ*

        # Save as tensors for fast indexing (these are what other methods will use)
        self._main_joint_ids = torch.tensor(main_indices, device=self.device, dtype=torch.long)
        self._finger_joint_ids = torch.tensor(finger_indices, device=self.device, dtype=torch.long)

        # Convenience alias for "non-finger" joints (used in _get_observations)
        self._non_finger_joint_ids = self._main_joint_ids

        # Sanity: we expect 27 DOFs total (15 main + 12 finger)
        assert len(self._main_joint_ids) == 15, f"Expected 15 main joints, got {len(self._main_joint_ids)}"
        assert len(self._finger_joint_ids) == 12, f"Expected 12 finger joints, got {len(self._finger_joint_ids)}"


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.sphere_object = RigidObject(self.cfg.sphere_cfg)
        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.target_object = RigidObject(self.cfg.target_cfg)
            self.scene.rigid_objects["target"] = self.target_object

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.rigid_objects["sphere"] = self.sphere_object
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self._apply_env_color_pairs()
        
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        sky_file = r"omniverse://localhost/NVIDIA/Assets/Skies/Indoor/ZetoCGcom_ExhibitionHall_Interior1.hdr"#http://omniverse-content-production.s3-us-west-2.amazonaws.com/Skies/Indoor/ZetoCG_com_WarehouseInterior2b.hdr"
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, #color=(0.7, 0.7, 0.7), \
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr")
        light_cfg.func("/World/Light", light_cfg, orientation=(1.0, 0.0, 0.0, 0.0))
        #spherelight_cfg = sim_utils.SphereLightCfg(intensity=2600.0, color=(1., 1.,1.), radius=1., treat_as_point=True)
        #spherelight_cfg.func("/World/envs/env_.*/Robot/Light", spherelight_cfg,  translation=(0.0, 0.0, 0.5))

    def _ensure_throw_hand_ids(self):
        """Cache palm + fingertip ids so we can respawn the ball in the hand."""
        if hasattr(self, "_throw_hand_body_id") and hasattr(self, "_right_fingertip_ids"):
            return

        body_names = list(self._robot.data.body_names)
        hand_side = getattr(self.cfg, "throw_hand_side", "right").lower()
        preferred_palm = "right_palm" if hand_side == "right" else "left_palm"
        try:
            self._throw_hand_body_id = body_names.index(preferred_palm)
        except ValueError:
            palm_candidates = [i for i, n in enumerate(body_names) if "palm" in n and hand_side in n]
            if len(palm_candidates) == 0:
                palm_candidates = [i for i, n in enumerate(body_names) if "palm" in n]
            assert len(palm_candidates) > 0, "No palm body found for Alpha"
            self._throw_hand_body_id = palm_candidates[-1]

        # Fingertip IDs based on hand side
        if hand_side == "right":
            fingertip_ids, _ = self._contact_sensor.find_bodies(".*right_(thumb|index|pinky)_distal.*")
        else:
            fingertip_ids, _ = self._contact_sensor.find_bodies(".*left_(thumb|index|pinky)_distal.*")
        if len(fingertip_ids) == 0:
            fingertip_ids, _ = self._contact_sensor.find_bodies(".*(thumb|index|pinky).*distal.*")
        self._right_fingertip_ids = fingertip_ids

    def _apply_env_color_pairs(self):
        """Assign a unique color to each env's ball/target pair for easier visual distinction."""
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

        color_indices = self._color_distribution_indices(len(env_colors))
        for env_index, env_path in enumerate(self.scene.env_prim_paths):
            if env_index >= len(env_colors):
                break
            color = env_colors[color_indices[env_index]]
            sphere_shader_path = f"{env_path}/sphere/geometry/{sphere_material_path}/Shader"
            self._set_shader_color(stage, sphere_shader_path, color)
            if target_material_path is not None:
                target_shader_path = f"{env_path}/target/geometry/{target_material_path}/Shader"
                self._set_shader_color(stage, target_shader_path, color)

    def _generate_env_color_palette(self, num_envs: int) -> list[tuple[float, float, float]]:
        """Generate evenly spaced colors on the HSV wheel for the requested number of environments."""
        if num_envs <= 0:
            return []
        saturation, value = 0.75, 0.9
        colors: list[tuple[float, float, float]] = []
        for env_index in range(num_envs):
            hue = (env_index / max(num_envs, 1)) % 1.0
            colors.append(self._hsv_to_rgb(hue, saturation, value))
        return colors


    @staticmethod
    def _set_shader_color(stage, shader_path: str, color: tuple[float, float, float]):
        prim = stage.GetPrimAtPath(shader_path)
        if prim.IsValid():
            sim_utils.safe_set_attribute_on_usd_prim(
                prim, "inputs:diffuseColor", color, camel_case=False
            )

    @staticmethod
    def _hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[float, float, float]:
        """Convert HSV color to RGB tuple."""
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
        """Return a permutation of indices so consecutive envs get well-separated colors."""
        if num_envs <= 1:
            return list(range(num_envs))
        step = max(1, num_envs // 3 or 1)
        # ensure step and num_envs are co-prime to cover all colors
        while gcd(step, num_envs) != 1:
            step += 1
        order = []
        idx = 0
        for _ in range(num_envs):
            order.append(idx)
            idx = (idx + step) % num_envs
        return order

    '''def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # copy actions to avoid in-place modification
        actions = actions.clone()
        ##########  TODO: Need to change on alpha  ##########
        limits = self._robot.data.joint_limits + self._robot.data.default_joint_pos.reshape((-1,1))
        # clip actions to be within the joint limits
        actions = torch.clamp(actions, limits[:, :, 0], limits[:,:, 1])
        return actions'''
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Optionally keep a copy for logging
        self.actions = actions.clone()

        # Base class usually sets self._actions; if not, do it here:
        self._actions = actions

        # Make sure joint index maps are built (from alpha_utils)
        self._build_alpha_joint_indices()
        main_idx = self._main_joint_ids           # [TJ1, RJ1..RJ7, LJ1..LJ7]
        finger_idx = self._finger_joint_ids       # 12 finger joints

        num_main = main_idx.shape[0]              # 15
        num_fingers = finger_idx.shape[0]         # 12
        num_dofs = self._robot.data.joint_pos.shape[1]

        # Sanity: expect main + finger = all DOFs
        assert num_main + num_fingers == num_dofs, (
            f"Alpha DOF mismatch: main={num_main}, fingers={num_fingers}, total={num_dofs}"
        )

        # Start from current joint positions (full DOF vector, in Isaac DOF order)
        current_all = self._robot.data.joint_pos.clone()
        self._processed_actions = current_all.clone()

        # ---------------- Main joints (torso + arms) ----------------
        # Interpret main actions as joint velocities (rad/s) and integrate to position targets.
        delta_main = self.cfg.action_scale * self._actions[:, :num_main] * self.step_dt
        self._processed_actions[:, main_idx] = current_all[:, main_idx] + delta_main

        # ---------------- Finger joints (gripper-like) ----------------
        # Actions layout:
        #   a[0 : num_main] -> torso + arms
        #   a[num_main]     -> grip scalar (open/close)
        grip_action = self._actions[:, num_main]  # shape: (num_envs,)

        # Closed-hand joint targets (rad): J1 = 0.7, J2 = 1.0
        closed_fingers = self._closed_finger_pose.unsqueeze(0).expand(self.num_envs, -1)
        open_fingers = torch.zeros_like(closed_fingers)

        # Grip rule: a > 0 → open, a <= 0 → closed (default closed so ball stays in hand)
        hand_close_mask = (grip_action <= 0).view(-1, 1)  # (num_envs, 1)

        finger_targets = torch.where(
            hand_close_mask,
            closed_fingers,   # closed
            open_fingers,     # open
        )  # (num_envs, 12)

        # Write finger targets into the correct DOF slots
        self._processed_actions[:, finger_idx] = finger_targets

        # Clip processed targets to joint limits (position and velocity-safe).
        pos_limits = getattr(self._robot.data, "joint_pos_limits", None)
        if pos_limits is not None:
            lower, upper = pos_limits[:, :, 0], pos_limits[:, :, 1]
            self._processed_actions = torch.clamp(self._processed_actions, lower, upper)

        # Clip commanded joint positions to the joint limits from sim data.
        if getattr(self._robot.data, "joint_pos_limits", None) is not None:
            limits = self._robot.data.joint_pos_limits  # (num_envs, num_dofs, 2)
            lower, upper = limits[:, :, 0], limits[:, :, 1]
            self._processed_actions = torch.clamp(self._processed_actions, lower, upper)


        # ------------------------------------------------------------
        # Optional: keep G1-style rendering hook
        # ------------------------------------------------------------
        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(torch.arange(self.num_envs, device=self.device))



    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._processed_actions)#[:,:-1])

    def _get_observations(self) -> dict:
        self._build_alpha_joint_indices()

        # import pdb; pdb.set_trace()

        ##########  Alpha-specific contact setup (regex already matches Alpha palms) ##########
        if not hasattr(self, "non_feet_ids"):
            # Alpha is fixed-base (no ankles), so this effectively selects all bodies.
            self.non_feet_ids, _ = self._contact_sensor.find_bodies("^(?!.*ankle).*$")

            # Palms + any finger geometries that still match these patterns.
            self.hand_ids, _ = self._contact_sensor.find_bodies(".*(palm|five|six|three|four|zero|one|two).*")

            # Explicit left/right palms (Alpha: left_palm / right_palm).
            self.left_hand_ids, _ = self._contact_sensor.find_bodies(".*left_(palm).*")
            self.right_hand_ids, _ = self._contact_sensor.find_bodies(".*right_(palm).*")

        ##########  Joint indices: use canonical torso + right arm + left arm ordering ##########
        if not hasattr(self, "_non_finger_joint_ids"):
            # Reuse the canonical main joint ordering from alpha_utils:
            # [TJ1, RJ1..RJ7, LJ1..LJ7]
            self._non_finger_joint_ids = self._main_joint_ids.to(self.device)

        # Joint state (exclude finger DOFs), in G1-compatible order
        idx = self._non_finger_joint_ids
        joint_pos_info = (
            self._robot.data.joint_pos[:, idx]
            - self._robot.data.default_joint_pos[:, idx]
        )
        joint_vel_info = self._robot.data.joint_vel[:, idx]

        ##########  Throw displacement estimation (same logic as G1) ##########
        estimated_displacement, estim_time = self.check_ball_displacement(
            torch.arange(self.num_envs, device=self.device)
        )
        estimated_displacement = 1 - estimated_displacement

        # Base orientation → roll (no IMU used)
        w, x, y, z = self._robot.data.root_quat_w.T
        roll = torch.atan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x**2 + y**2)
        ).to(self.device)

        # Curriculum-style noisy privileged signal
        noise_displace = (
            (1 - self.action_noise) * estimated_displacement
            + self.action_noise * torch.randn_like(estimated_displacement)
        )

        # Store base velocity if used elsewhere (not part of the observation IMU-wise)
        self.prev_velocity = self._robot.data.root_lin_vel_b.clone()

        # Ball linear velocity (world frame): vx, vy, vz
        ball_data = self.sphere_object.data.body_state_w[:, 0, [7, 8, 9]]

        ##########  Assemble observation vector ##########
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    # No linear IMU term here.
                    self._robot.data.root_ang_vel_b if self.cfg.obs_ang_vel else None,
                    self._robot.data.projected_gravity_b if self.cfg.obs_proj_grav else None,
                    (roll.float() + (torch.rand_like(roll) * 0.02 - 0.01)).unsqueeze(-1)
                    if self.cfg.obs_roll
                    else None,
                    # Throw target commands: (θ̃, φ, r)
                    self.throwing_commands,
                    # Joint state (non-finger) with small noise
                    joint_pos_info + (torch.rand_like(joint_pos_info) * 0.02 - 0.01),
                    joint_vel_info + (torch.rand_like(joint_vel_info) * 0.1 - 0.05),
                    # Previous action (full, including fingers)
                    self._actions,
                    # Ball released flag (1 = not released, 0 = released)
                    self.not_released_ball.unsqueeze(-1).float()
                    if self.cfg.obs_notrelease
                    else None,
                    # Privileged estimated displacement + time + ball vel (optionally)
                    noise_displace.unsqueeze(-1)
                    if self.cfg.obs_estimdisplace
                    else None,
                    estim_time.unsqueeze(-1)
                    if self.cfg.obs_estimdisplace
                    else None,
                    ball_data
                    if self.cfg.obs_estimdisplace
                    else None,
                )
                if tensor is not None
            ],
            dim=-1,
        )

        obs = torch.clip(obs, -1000, 1000)
        observations = {"policy": obs}

        if torch.sum(torch.isnan(obs)) > 0:
            print(torch.sum(torch.isnan(obs), dim=0))

        return observations


    def _get_rewards(self) -> torch.Tensor:
        """Event-based throwing reward with mean L2 penalties and extra debug logs."""
        self._ensure_throw_hand_ids()

        # Main/finger index tensors for penalties
        if not hasattr(self, "_non_finger_joint_ids"):
            self._build_alpha_joint_indices()
            self._non_finger_joint_ids = self._main_joint_ids.to(self.device)
        idx = self._non_finger_joint_ids

        # --- Release / throwing reward -------------------------------------------------
        ball_pos = self.sphere_object.data.body_pos_w[:, 0, 0:3]
        hand_pos = self._robot.data.body_pos_w[:, self._throw_hand_body_id, :]
        dist = torch.norm(ball_pos - hand_pos, dim=1)

        release_mask = (dist > 0.25) & (~self.throwing_reward_given)
        self.not_released_ball &= ~release_mask
        ball_released_envs = torch.nonzero(release_mask, as_tuple=False).flatten()

        self.throwing_reward.zero_()
        vel_align_rew = torch.zeros(self.num_envs, device=self.device)

        if ball_released_envs.numel() > 0:
            disp, landing_time = self.check_ball_displacement(ball_released_envs)
            self.throwing_reward[ball_released_envs] = torch.clamp(1.0 - disp, min=0.0)
            self.throwing_reward_given[ball_released_envs] = True
            self.landing_time[ball_released_envs] = landing_time + self.episode_length_buf[ball_released_envs] * self.step_dt
            self.release_ball_pos[ball_released_envs] = ball_pos[ball_released_envs]
            target_pos = self.target_positions[ball_released_envs] + self._terrain.env_origins[ball_released_envs]
            self.release_target_dir[ball_released_envs] = target_pos - ball_pos[ball_released_envs]

        # Velocity / direction alignment each step for released balls
        released_mask = ~self.not_released_ball
        if released_mask.any():
            released_ids = torch.nonzero(released_mask, as_tuple=False).flatten()
            ball_vel = self.sphere_object.data.body_state_w[released_ids, 0, 7:10]
            target_pos = self.target_positions[released_ids] + self._terrain.env_origins[released_ids]
            target_dir = target_pos - ball_pos[released_ids]
            dot = torch.sum(ball_vel * target_dir, dim=1)
            denom = torch.clamp(torch.norm(ball_vel, dim=1) * torch.norm(target_dir, dim=1), min=1e-6)
            cos_sim = torch.clamp(dot / denom, -1.0, 1.0)
            vel_align_rew[released_ids] = torch.clamp(cos_sim, min=0.0)

        # --- Stability / collisions ----------------------------------------------------
        ball_positions = self.sphere_object.data.root_pos_w.clone()
        ball_not_thrown_cond = (torch.norm((hand_pos - ball_positions), dim=1) <= 0.25) & (self.reset_buf == 1)

        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)
        mask = torch.where(
            torch.norm((hand_pos - ball_positions), dim=1) <= 0.25,
            torch.zeros(1, device=self.device),
            torch.ones(1, device=self.device),
        )
        if hasattr(self, "hand_ids"):
            first_contact[:, self.hand_ids] &= mask.view(-1, 1).expand(-1, len(self.hand_ids)).to(torch.bool)
        non_feet = getattr(self, "non_feet_ids", None)
        if non_feet is None:
            non_feet = torch.arange(first_contact.shape[1], device=self.device)
        collision = torch.sum(first_contact[:, non_feet], dim=1) > 0

        self.stability_penalty_this_ep |= (ball_not_thrown_cond | collision)
        if getattr(self.cfg, "nonsparse_stability_reward", False):
            stability_rew = ((~(ball_not_thrown_cond | collision)).float() / self.max_episode_length_s) * self.step_dt
        else:
            stability_rew = ((self.reset_buf == 1) & (~self.stability_penalty_this_ep)).float()

        # --- Smoothness penalties (mean over main joints) ------------------------------
        action_rate = torch.mean(torch.square(self._actions[:, : idx.shape[0]] - self._previous_actions[:, : idx.shape[0]]), dim=1)
        joint_torques = torch.mean(torch.square(self._robot.data.applied_torque[:, idx]), dim=1)
        # Disable velocity/acceleration penalty for now
        joint_accel = torch.zeros_like(action_rate)

        # --- Ball release bonus (event-based, single step) -----------------------------
        ballrel_rew = torch.zeros(self.num_envs, device=self.device)
        if ball_released_envs.numel() > 0:
            ballrel_rew[ball_released_envs] = getattr(self.cfg, "ball_release_reward_scale", 0.0)

        # --- Rewards dictionary (order matches reward_component_names) -----------------
        rewards = {
            "throwing": torch.clamp(self.throwing_reward, min=0.0) * self.cfg.throwing_reward_scale,
            "stability": stability_rew * self.cfg.stability_reward_scale,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "ballrel_rew": ballrel_rew,
            "vel_align_rew": vel_align_rew * getattr(self.cfg, "vel_align_reward_scale", 0.0),
        }

        # Logging / debugging
        if not hasattr(self, "extras"):
            self.extras = {}
        self.extras["log"] = self.extras.get("log", {})
        n_env = float(self.num_envs)
        self.extras["log"]["debug/collision_pct"] = 100.0 * collision.float().mean()
        self.extras["log"]["debug/ball_not_thrown_pct"] = 100.0 * ball_not_thrown_cond.float().mean()
        self.extras["log"]["debug/stability_penalty_pct"] = 100.0 * self.stability_penalty_this_ep.float().mean()
        self.extras["log"]["debug/ball_released_pct"] = 100.0 * (ball_released_envs.numel() / max(n_env, 1.0))

        # Accumulate episodic sums
        for key, value in rewards.items():
            self._episode_sums[key] += value

        # Keep previous actions after reward is computed (for next step penalty)
        self._previous_actions = self._actions.clone()

        # Stack rewards in the predefined component order
        reward_vec = torch.stack([rewards[name] for name in self.reward_component_names], dim=1)
        return reward_vec

    def _check_joint_velocity_violation(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect excessive joint velocities on main joints and accumulate violation flags."""
        limit = getattr(self.cfg, "joint_velocity_limit", None)
        penalty_scale = getattr(self.cfg, "joint_velocity_penalty_scale", 0.0)
        if limit is None:
            violation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return violation, torch.zeros(self.num_envs, device=self.device)

        joint_vel = self._robot.data.joint_vel[:, self._main_joint_ids]
        violation = torch.any(torch.abs(joint_vel) > limit, dim=1)
        self._joint_vel_violation |= violation
        penalty = violation.float() * penalty_scale
        return self._joint_vel_violation.clone(), penalty


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # World-frame ball + target
        ball_pos = self.sphere_object.data.root_pos_w  # (N, 3)
        target_pos = self.target_positions + self._terrain.env_origins  # (N, 3)

        released = ~self.not_released_ball
        hit_ground = (ball_pos[:, 2] < 0.05) & released  # after release, ball on ground
        too_far = released & (torch.norm(ball_pos[:, :2] - target_pos[:, :2], dim=1) > self.cfg.max_throw_dist + 1.0)
        joint_vel_violation, _ = self._check_joint_velocity_violation()

        # Optional: no release within 75% of the episode
        timeout_frac = getattr(self.cfg, "no_release_timeout_frac", 0.75)
        no_release_timeout = (self.episode_length_buf > timeout_frac * self.max_episode_length) & (~released)

        terminated = hit_ground | too_far | no_release_timeout | joint_vel_violation

        # Success flag: ball close to target after landing
        success = hit_ground & (torch.norm(ball_pos - target_pos, dim=1) < 0.3)
        self.extras["termination"] = {
            "terminated": terminated,
            "truncated": time_out,
            "success": success,
            "joint_vel_violation": joint_vel_violation,
        }

        return terminated, time_out


    # def _reset_idx(self, env_ids: Sequence[int] | None):
    #     if env_ids is None or len(env_ids) == self.num_envs:
    #         env_ids = self._robot._ALL_INDICES
        
    #     self._robot.reset(env_ids)
    #     super()._reset_idx(env_ids)
        

    #     if self.cfg.arm_only:
    #         self.body_actions = 0.
    #     else:
    #         self.body_actions = 1. # max is 100%

    #     if self.cfg.distance_throw:
    #         self.distance_range = [max(self.distance_range[0], 4), max(self.distance_range[1], 4)]
        
    #     if self.cfg.no_proj_motion:
    #         self.cfg.obs_estimdisplace = False



    #     self._actions[env_ids] = 0.0
    #     self._previous_actions[env_ids] = 0.0

    #     self._sample_throwing_commands(env_ids)

    #     self.target_positions[env_ids] = self.calculate_target_offset(env_ids)
    #     self.throwing_reward[env_ids] = -1 
    #     self.landing_time[env_ids] = -1
    #     self.throwing_reward_given[env_ids] = False
    #     self.not_released_ball[env_ids] = True
    #     self.sum_open_hand_action[env_ids] = 0
    #     self.released_ball_t[env_ids] = -1
    #     # self.stability_penalty_this_ep[env_ids] = False # removed on purpose


    #     ##### TODO: Need to change on alpha  ##########
    #     # Reset robot state
    #     joint_pos = self._robot.data.default_joint_pos[env_ids] 
    #     #joint_pos[:,[1,3,5, 7,9, 11]] += torch.zeros_like(joint_pos[:,[1,3,5, 7,9, 11]]).uniform_(-self.cfg.arm_dr_range, self.cfg.arm_dr_range) # hand .uniform_(-1,1)
    #     joint_pos[:,[5,6,9,10,13,14,17,18,21,22]] += torch.zeros_like(joint_pos[:,[5,6,9,10,13,14,17,18,21,22]]).uniform_(-0.3, 0.3) # hand .uniform_(-1,1)
    #     joint_pos += torch.zeros_like(joint_pos).uniform_(-0.05, 0.05)

    #     joint_vel = self._robot.data.default_joint_vel[env_ids]
    #     default_root_state = self._robot.data.default_root_state[env_ids]
    #     if self.robot_yaw_offset_rad != 0.0:
    #         offset_w = math.cos(self.robot_yaw_offset_rad * 0.5)
    #         offset_z = math.sin(self.robot_yaw_offset_rad * 0.5)
    #         offset_quat = torch.zeros_like(default_root_state[:, 3:7])
    #         offset_quat[:, 0] = offset_w
    #         offset_quat[:, 3] = offset_z
    #         default_root_state[:, 3:7] = self._quat_multiply(offset_quat, default_root_state[:, 3:7])
    #     default_root_state[:, :3] += self._terrain.env_origins[env_ids]

    #     self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    #     self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
    #     self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    #     ##############################################################

    #     if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
    #         self.initialise_target_for_rendering(env_ids)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # ----------------- Standard IsaacLab reset plumbing ----------------- #
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # ----------------- Body action scaling (unchanged) ------------------ #
        if self.cfg.arm_only:
            self.body_actions = 0.0
        else:
            self.body_actions = 1.0  # max is 100%

        # Distance-throw specific clamp (unchanged)
        if self.cfg.distance_throw:
            self.distance_range = [
                max(self.distance_range[0], 4),
                max(self.distance_range[1], 4),
            ]

        # Disable privileged displacement obs if we don't use projected motion
        if self.cfg.no_proj_motion:
            self.cfg.obs_estimdisplace = False

        # ----------------- Clear action-related buffers --------------------- #
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._build_alpha_joint_indices()
        grip_action_index = self._main_joint_ids.shape[0]
        self._actions[env_ids, grip_action_index] = -1.0
        self._previous_actions[env_ids, grip_action_index] = -1.0

        # ----------------- Sample new target / commands --------------------- #
        self._sample_throwing_commands(env_ids)
        self.target_positions[env_ids] = self.calculate_target_offset(env_ids)

        # Throw metrics
        self.throwing_reward[env_ids] = -1 
        self.landing_time[env_ids] = -1
        self.throwing_reward_given[env_ids] = False
        self.target_hit_given[env_ids] = False
        self.not_released_ball[env_ids] = True
        self.sum_open_hand_action[env_ids] = 0
        self.released_ball_t[env_ids] = -1
        self._joint_vel_violation[env_ids] = False
        # self.stability_penalty_this_ep[env_ids] = False  # intentionally left as-is

        # -------------------------------------------------------------------- #
        #                  Alpha-specific joint state reset
        #   (Replace G1 magic indices [5,6,9,10,13,14,17,18,21,22] etc.)
        # -------------------------------------------------------------------- #

        # Ensure canonical joint index mapping (torso + arms + fingers) exists
        if not hasattr(self, "main_joint_indices") or not hasattr(self, "finger_joint_indices"):
            self._build_alpha_joint_indices()

        # Start from default joint configuration
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        # 1) Rotate torso yaw (first main joint) to face the throw side:
        #    right-hand throw -> -90 deg, left-hand throw -> +90 deg.
        hand_side = getattr(self.cfg, "throw_hand_side", "right").lower()
        if hand_side == "right":
            joint_pos[:, self.main_joint_indices[0]] -= math.pi / 2.0
        else:
            joint_pos[:, self.main_joint_indices[0]] += math.pi / 2.0

        # 2) Randomize torso + arm joints (main joints) within arm_dr_range
        #    This replaces the old G1 hand/arm jitter by specific indices.
        if getattr(self.cfg, "arm_dr_range", 0.0) > 0.0:
            rand_main = torch.zeros_like(joint_pos[:, self.main_joint_indices])
            rand_main.uniform_(-self.cfg.arm_dr_range, self.cfg.arm_dr_range)
            joint_pos[:, self.main_joint_indices] += rand_main

        # 3) Fingers: start closed so the ball is held firmly at reset.
        if hasattr(self, "finger_joint_indices"):
            joint_pos[:, self.finger_joint_indices] = self._closed_finger_pose

        # 4) Small global noise on all DOFs (same spirit as G1: +/- 0.05)
        joint_pos += torch.zeros_like(joint_pos).uniform_(-0.05, 0.05)
        if hasattr(self, "finger_joint_indices"):
            joint_pos[:, self.finger_joint_indices] = self._closed_finger_pose

        # -------------------------------------------------------------------- #
        #                  Root pose / velocity (mostly unchanged)
        #   Alpha is fixed-base, but we still support yaw offsets and env_origins
        #   for multiple parallel envs.
        # -------------------------------------------------------------------- #
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        if self.robot_yaw_offset_rad != 0.0:
            offset_w = math.cos(self.robot_yaw_offset_rad * 0.5)
            offset_z = math.sin(self.robot_yaw_offset_rad * 0.5)
            offset_quat = torch.zeros_like(default_root_state[:, 3:7])
            offset_quat[:, 0] = offset_w
            offset_quat[:, 3] = offset_z
            default_root_state[:, 3:7] = self._quat_multiply(
                offset_quat, default_root_state[:, 3:7]
            )

        # Shift each env to its origin tile
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # Write everything back into sim
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._reset_ball_to_hand(env_ids)

        # -------------------------------------------------------------------- #
        #                      Target visualization (unchanged)
        # -------------------------------------------------------------------- #
        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(env_ids)


    def _reset_ball_to_hand(self, env_ids: Sequence[int] | torch.Tensor):
        """Respawn the ball in the throwing hand so each episode starts with a grasp."""
        if env_ids is None or len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._ensure_throw_hand_ids()

        ball_state = self.sphere_object.data.default_root_state.clone()[env_ids]
        hand_pos = self._robot.data.body_pos_w[env_ids, self._throw_hand_body_id]
        hand_quat = self._robot.data.body_quat_w[env_ids, self._throw_hand_body_id]
        hand_vel = self._robot.data.body_vel_w[env_ids, self._throw_hand_body_id]

        # Default: place ball slightly farther out of the palm to avoid initial intersections
        forward_offset_local = torch.tensor([0.0, 0.08, 0.04], device=self.device)
        forward_offset_world = self._quat_apply(hand_quat, forward_offset_local.expand(len(env_ids), -1))
        ball_offset = forward_offset_world

        # If fingertips are available, bias placement between palm and fingertip cluster
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
                ball_offset[mask] = dir_norm * 0.05  # push ~5 cm toward fingertips

        ball_pos = hand_pos + ball_offset
        ball_state[:, :3] = ball_pos
        ball_state[:, 3:7] = self.sphere_object.data.default_root_state[env_ids, 3:7]
        ball_state[:, 7:] = hand_vel.reshape(len(env_ids), 6)
        self.sphere_object.write_root_state_to_sim(ball_state, env_ids)


    def update_curriculum(self, iter):
        log_dict = self.extras.get("log", {}) if hasattr(self, "extras") else {}
        stability_value = log_dict.get("Episode_Reward/stability", None)
        throwing_value = log_dict.get("Episode_Reward/throwing", None)
        if stability_value is None or throwing_value is None:
            return  # nothing to update yet

        self.action_noise = min(1., self.action_noise + 0.001)

        if (throwing_value > self.cfg.r_throw_thresh and stability_value > self.cfg.r_stability_thresh) or \
            (iter >= 5000):
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
                self.theta_range = [min(0., self.theta_range[0]), min(1.0, self.theta_range[1] + 0.01)]
            new_height_max = min(self.cfg.target_height_range[1], self.current_target_height_range[1] + height_step)
            self.current_target_height_range[1] = new_height_max

    def _sample_throwing_commands(self, env_ids: torch.Tensor | None):
        """Sample target distance/angles within a forward FOV and height band."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        command_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Azimuth within forward cone
        if self.target_half_fov_rad <= 0.0:
            phi_samples = torch.zeros_like(self.throwing_commands[command_ids, 2])
        else:
            hand_side = getattr(self.cfg, "throw_hand_side", "right").lower()
            if hand_side == "right":
                phi_low, phi_high = 0.0, self.target_half_fov_rad
            else:
                phi_low, phi_high = -self.target_half_fov_rad, 0.0
            phi_samples = torch.zeros_like(self.throwing_commands[command_ids, 2]).uniform_(phi_low, phi_high)

        # Fixed-target overrides for overfitting/debug. Set cfg.fixed_* to use constants.
        fixed_dist = getattr(self.cfg, "fixed_throw_dist", None)
        fixed_height = getattr(self.cfg, "fixed_target_height", None)

        if fixed_dist is not None:
            dist_final = torch.full_like(self.throwing_commands[command_ids, 0], float(fixed_dist))
        else:
            dist_min, dist_max = self.distance_range
            margin = 0.05
            dist_samples = torch.zeros_like(self.throwing_commands[command_ids, 0]).uniform_(dist_min, dist_max)
            dist_final = dist_samples

        if fixed_height is not None:
            z_clamped = torch.full_like(self.throwing_commands[command_ids, 0], float(fixed_height))
        else:
            z_min, z_max = self.current_target_height_range
            margin = 0.05
            z_samples = torch.zeros_like(self.throwing_commands[command_ids, 0]).uniform_(z_min, z_max)
            z_clamped = torch.clamp(z_samples, min=0.0, max=dist_final.max() - margin)

        theta_cos = torch.clamp(z_clamped / torch.clamp(dist_final, min=1e-3), -1.0, 1.0)

        if self.cfg.distance_throw:
            # Keep previous distance-throw semantics (purely forward, flat)
            theta_cos = torch.ones_like(theta_cos)
            phi_samples = torch.zeros_like(phi_samples)

        self.throwing_commands[command_ids, 0] = dist_final
        self.throwing_commands[command_ids, 1] = theta_cos
        self.throwing_commands[command_ids, 2] = phi_samples

    def _get_base_yaw(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Extract the base yaw for each env index."""
        command_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        base_quats = self._robot.data.root_quat_w[command_ids]
        w, x, y, z = base_quats.unbind(dim=1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.where(torch.isnan(yaw), torch.zeros_like(yaw), yaw)

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions (w, x, y, z)."""
        w1, x1, y1, z1 = q1.unbind(dim=1)
        w2, x2, y2, z2 = q2.unbind(dim=1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack((w, x, y, z), dim=1)

    @staticmethod
    def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate vector(s) by quaternion(s)."""
        w, x, y, z = quat.unbind(dim=1)
        vx, vy, vz = vec.unbind(dim=1)
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rx = (ww + xx - yy - zz) * vx + 2 * ((xy - wz) * vy + (xz + wy) * vz)
        ry = 2 * ((xy + wz) * vx + (ww - xx + yy - zz) * vy + (yz - wx) * vz)
        rz = 2 * ((xz - wy) * vx + (yz + wx) * vy + (ww - xx - yy + zz) * vz)
        return torch.stack((rx, ry, rz), dim=1)

    def calculate_target_offset(self, env_ids: torch.Tensor) -> torch.Tensor:
        command_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        dist = self.throwing_commands[command_ids, 0]
        theta_cos = torch.clamp(self.throwing_commands[command_ids, 1], -1.0, 1.0)
        phi_body = self.throwing_commands[command_ids, 2]
        theta = torch.arccos(theta_cos)

        x_local = dist * torch.sin(theta) * torch.cos(phi_body)
        y_local = dist * torch.sin(theta) * torch.sin(phi_body)
        z_local = dist * torch.cos(theta)

        base_yaw = self._get_base_yaw(command_ids)
        yaw = base_yaw + self.target_heading_offset_rad
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        x_world = x_local * cos_yaw - y_local * sin_yaw
        y_world = x_local * sin_yaw + y_local * cos_yaw

        res = torch.stack((x_world, y_world, z_local), dim=1)
        return res.squeeze(-1)
    
    def check_ball_displacement(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ball_data = self.sphere_object.data.body_state_w[env_ids, 0, :10]
        target_positions = self.target_positions[env_ids] + self._terrain.env_origins[env_ids] 
        target_positions[:,:2] += self.default_root_states[env_ids,:2] #self.scene.env_origins[env_ids]

        if self.cfg.air_resistance:
            # handle air resistance case if needed...
            raise NotImplementedError
        else:
            # initial velocities
            vx0 = ball_data[:, 7]
            vy0 = ball_data[:, 8]
            vz0 = ball_data[:, 9]
            a = 9.81

            # initial height minus 3cm floor offset
            z0 = ball_data[:, 2]
            z0_minus_0_03 = torch.clamp(z0 - 0.03, min=0.0)
            sqrt_term = torch.sqrt(vz0**2 + 2 * a * z0_minus_0_03)

            # times to hit the floor
            t1 = (-vz0 + sqrt_term) / -a
            t2 = (-vz0 - sqrt_term) / -a
            tz = torch.max(t1, t2)  # shape: (batch,)
            tz = torch.clamp(tz, min=0.0)

            # build a time-grid [steps x batch]
            steps = 100
            frac = torch.linspace(0, 1, steps=steps, device=self.device).unsqueeze(1)  # [steps, 1]
            time_tensor = frac * tz.unsqueeze(0)  # [steps, batch]

            # compute trajectories at each time
            new_x = ball_data[:, 0].unsqueeze(0) + vx0.unsqueeze(0) * time_tensor
            new_y = ball_data[:, 1].unsqueeze(0) + vy0.unsqueeze(0) * time_tensor
            new_z = (ball_data[:, 2].unsqueeze(0) +
                    vz0.unsqueeze(0) * time_tensor -
                    0.5 * a * time_tensor**2)

            # normalized distance to target
            distance_command = self.throwing_commands[env_ids, 0].unsqueeze(0)  # [1, batch]
            disp_matrix = torch.sqrt(
                (new_x - target_positions[:, 0].unsqueeze(0))**2 +
                (new_y - target_positions[:, 1].unsqueeze(0))**2 +
                (new_z - target_positions[:, 2].unsqueeze(0))**2
            ) / distance_command
            disp_matrix = torch.clamp(disp_matrix, max=1.0)

            # find minimum displacement and its index along the time axis
            disp_min, idx_min = torch.amin(disp_matrix, dim=0, keepdim=False), torch.argmin(disp_matrix, dim=0)

            # gather the corresponding time for each env
            batch_idx = torch.arange(len(env_ids), device=self.device)
            time_at_min = time_tensor[idx_min, batch_idx]  # [batch,]

        #print(torch.mean(time_at_min), torch.amin(time_at_min), torch.amax(time_at_min), torch.std(time_at_min))
        return disp_min, time_at_min

    def initialise_target_for_rendering(self, env_ids):
        target_default_state = self.target_object.data.default_root_state.clone()[env_ids]
        target_default_state[:, 7:] = torch.zeros_like(self.target_object.data.default_root_state[env_ids, 7:])
        target_default_state[:, 0:3] += self.target_positions[env_ids] + self._terrain.env_origins[env_ids] 
        target_default_state[:,:2] += self.default_root_states[env_ids,:2] #self.scene.env_origins[env_ids]

        # Compute the direction vector from current position to target position (offset -1,0,0.3)
        direction_to_target = self._robot.data.default_root_state[env_ids, :3] - (self.target_positions[env_ids]) 

        # Assuming the forward vector of the object is [1, 0, 0]
        forward_vector = torch.tensor([[1, 0, 0]], device=target_default_state.device).expand(direction_to_target.size(0), -1).float()

        direction_to_target = F.normalize(direction_to_target, p=2, dim=1)

        # Calculate the cross product to get the axis of rotation
        axis_of_rotation = torch.cross(forward_vector, direction_to_target, dim=1)

        # Calculate the angle between the forward vector and the direction to the target
        dot_product = (forward_vector * direction_to_target).sum(dim=1, keepdim=True)
        # Ensure angle has shape [batch_size, 1]
        angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))  # Shape: [batch_size, 1]

        # Apply sin to the angle and ensure it has the correct shape
        sin_half_angle = torch.sin(angle / 2)  # Shape: [batch_size, 1]

        # Multiply the axis of rotation by sin(angle/2) (element-wise)
        xyz = axis_of_rotation * sin_half_angle  # Shape: [batch_size, 3]

        # w is cos(angle/2), ensuring it has shape [batch_size, 1]
        w = torch.cos(angle / 2)  # Shape: [batch_size, 1]

        # Now concatenate w and xyz to form the quaternion, which will have shape [batch_size, 4]
        quaternion = torch.cat([w, xyz], dim=1)  # Shape: [batch_size, 4]
        current_quaternion = target_default_state[:, 3:7]
        # Now apply the quaternion to rotate the object (this depends on how your system applies rotations)
        # You may want to convert the quaternion into a rotation matrix for further usage
        # Function to multiply two quaternions
        def quaternion_multiply(q1, q2):
            w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
            w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
            
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return torch.stack([w, x, y, z], dim=1)

        # Update the quaternion by multiplying the current one with the new rotation quaternion
        new_quaternion = quaternion_multiply(current_quaternion, quaternion)

        # Set the updated orientation back into the default state
        target_default_state[:, 3:7] = new_quaternion
        self.target_object.write_root_state_to_sim(target_default_state, env_ids)
