# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from math import gcd
from typing import Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F
import isaacsim.core.utils.stage as stage_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .throwing_env_cfg import ThrowingGeneralEnvCfg


class ThrowingEnv(DirectRLEnv):
    """Whole-body throwing environment with standing stability + velocity-shaping."""

    cfg: ThrowingGeneralEnvCfg

    def __init__(self, cfg: ThrowingGeneralEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ------------------------------------------------------------------
        # Reward scales (keep these similar magnitude so nothing dominates)
        # ------------------------------------------------------------------
        self.throwing_reward_scale = getattr(self.cfg, "throwing_reward_scale", 5.0)
        self.roll_reward_scale = getattr(self.cfg, "roll_reward_scale", 0.3)
        self.stability_reward_scale = getattr(self.cfg, "stability_reward_scale", 0.25)
        self.action_rate_reward_scale = getattr(self.cfg, "action_rate_reward_scale", -1e-3)
        self.joint_torque_reward_scale = getattr(self.cfg, "joint_torque_reward_scale", -2.5e-6)
        self.joint_accel_reward_scale = getattr(self.cfg, "joint_accel_reward_scale", -2.5e-8)

        # make shaping genuinely "shaping", not the main objective
        self.vel_tracking_reward_scale = getattr(self.cfg, "vel_tracking_reward_scale", 0.3)
        self.fingers_not_blocking_reward_scale = getattr(self.cfg, "fingers_not_blocking_reward_scale", 0.3)

        # Base-height / fall calibration (robot-agnostic; set on first reset)
        self.min_base_height: float | None = None
        self.fall_margin: float = 0.20  # meters below default standing height considered "fall"

        # Actions
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions = torch.zeros_like(self._actions)

        # Throwing commands (distance, cos(theta), phi)
        self.throwing_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_positions = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging per-episode reward components
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "throwing",
                "roll",
                "stability",
                "action_rate_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "vel_tracking",
                "fingers_not_blocking",
            ]
        }
        self.reward_components = len(self._episode_sums.keys())
        self.reward_component_names = list(self._episode_sums.keys())
        self.reward_component_task_rew = [
            "throwing",
            "roll",
            "stability",
            "vel_tracking",
            "fingers_not_blocking",
        ]

        # Throwing-related buffers
        self.not_released_ball = torch.ones((self.num_envs), device=self.device, dtype=torch.bool)
        self.throwing_reward = torch.ones((self.num_envs), device=self.device) * -1.0
        self.landing_time = torch.ones((self.num_envs), device=self.device) * -1.0
        self.throwing_reward_given = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        self.stability_penalty_this_ep = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        self.sum_open_hand_action = torch.zeros((self.num_envs), device=self.device)
        self.stability_timer = 2.0  # seconds

        # Distance / angle curriculum
        if self.cfg.distance_throw:
            self.distance_range = [4.0, 4.0]
            self.theta_range = [1.0, 1.0]
        else:
            self.distance_range = [4.0, 8.0]
            self.theta_range = [0.0, 1.0]

        self.target_half_fov_rad = min(math.radians(self.cfg.target_fov_deg) / 2.0, math.pi)
        self.target_heading_offset_rad = math.radians(self.cfg.target_heading_offset_deg)
        self.robot_yaw_offset_rad = math.radians(self.cfg.robot_yaw_offset_deg)

        # Sample initial throwing commands
        self._sample_throwing_commands(torch.arange(self.num_envs, device=self.device))

        self.released_ball_t = torch.zeros((self.num_envs), device=self.device) - 1.0
        self.action_noise = 0.0

        self.ball_distances = torch.zeros((self.num_envs), device=self.device) - 1.0

        if self.cfg.no_proj_motion:
            self.cfg.obs_estimdisplace = False

        self.prev_velocity = torch.zeros((self.num_envs, 3), device=self.device) - 1000.0

        # world-space root position offsets (for projectile calcs)
        self.default_root_states = torch.zeros((self.num_envs, 3), device=self.device)

        # contact-related caches (filled lazily)
        self.non_feet_ids = None
        self.hand_ids = None
        self._hip_ids = None
        self.left_hand_ids = None
        self.right_hand_ids = None

    # --------------------------------------------------------------------------
    # Scene setup
    # --------------------------------------------------------------------------
    def _setup_scene(self):
        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Ball
        self.sphere_object = RigidObject(self.cfg.sphere_cfg)
        self.scene.rigid_objects["sphere"] = self.sphere_object

        # Target (only needed when rendering)
        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.target_object = RigidObject(self.cfg.target_cfg)
            self.scene.rigid_objects["target"] = self.target_object

        # Contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone, filter, replicate
        self.scene.clone_environments(copy_from_source=False)
        self._apply_env_color_pairs()

        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lights
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                "kloofendal_43d_clear_puresky_4k.hdr"
            ),
        )
        light_cfg.func("/World/Light", light_cfg, orientation=(1.0, 0.0, 0.0, 0.0))

    # --------------------------------------------------------------------------
    # Visualization helpers
    # --------------------------------------------------------------------------
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
            sim_utils.safe_set_attribute_on_usd_prim(prim, "inputs:diffuseColor", color, camel_case=False)

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

    # --------------------------------------------------------------------------
    # RL interface
    # --------------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        # Store actions
        self._actions = actions.clone()

        # Non-hand joints: we assume first 23 entries are non-hand (body + arm)
        default_positions = self._robot.data.default_joint_pos[
            :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        ].clone()
        scalar_factor = 3.0
        self._processed_actions = self.cfg.action_scale * self._actions[:, :-1] + default_positions

        # Finger actions (Unitree G1 hand layout)
        finger_positions = torch.tensor(
            [-0.35, -0.61, 0.0, 0.61, 0.35, 0.61, 0.35],
            device=self.device,
        ) * scalar_factor
        finger_positions[1] /= 2.0  # example scaling for a two-link finger
        finger_positions[0] /= 2.0

        # Full 14 finger joints (indices -14:-1 in original layout)
        full_finger_actions = torch.zeros((self.num_envs, 14), device=self.device)

        # Map finger positions to relative indices (tuned for G1 joint order)
        finger_indices = [-3, -1, -9, -4, -10, -5, -11]  # original relative indices
        array_indices = [14 + idx for idx in finger_indices]  # convert to 0-based

        for i, pos_idx in enumerate(array_indices):
            full_finger_actions[:, pos_idx] = finger_positions[i]

        # Hand open/close mask from last action dimension
        hand_open_mask = self._actions[:, -1] >= 0.0
        full_finger_actions[~hand_open_mask] = 0.0

        # Concatenate processed actions
        self._processed_actions = torch.cat([self._processed_actions, full_finger_actions], dim=1)

        # Initialize targets for rendering (optional)
        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(torch.arange(self.num_envs, device=self.device))

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        # Lazily set contact-related ids once, using your actual link names
        if self.non_feet_ids is None:
            # Everything except ankle links
            self.non_feet_ids, _ = self._contact_sensor.find_bodies("^(?!.*ankle).*$")

            # All hand-related links: palms + finger segments
            self.hand_ids, _ = self._contact_sensor.find_bodies(
                ".*(palm|five|six|three|four|zero|one|two).*"
            )

            # Hip / pelvis
            self._hip_ids, _ = self._contact_sensor.find_bodies(".*pelvis.*")

            # Left and right palm specifically
            self.left_hand_ids, _ = self._contact_sensor.find_bodies("left_palm_link")
            self.right_hand_ids, _ = self._contact_sensor.find_bodies("right_palm_link")

        self._previous_actions = self._actions.clone()

        # Joint info (exclude hand joints)
        joint_pos_info = (self._robot.data.joint_pos - self._robot.data.default_joint_pos)[:, :23]
        joint_vel_info = self._robot.data.joint_vel[:, :23]

        # Projectile-based estimated displacement/time to target
        estimated_displacement, estim_time = self.check_ball_displacement(
            torch.arange(self.num_envs, device=self.device)
        )
        estimated_displacement = 1.0 - estimated_displacement

        # Base roll
        w, x, y, z = self._robot.data.root_quat_w.T
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2)).to(self.device)
        roll = torch.where(torch.isnan(roll), torch.zeros_like(roll), roll)

        # Add small noise to displacement (for robustness)
        noise_displace = (1.0 - self.action_noise) * estimated_displacement + self.action_noise * torch.randn_like(
            estimated_displacement
        )

        self.prev_velocity = self._robot.data.root_lin_vel_b.clone()

        # Ball velocities (vx, vy, vz)
        ball_data = self.sphere_object.data.body_state_w[:, 0, [7, 8, 9]]

        obs_tensors = [
            self._robot.data.root_ang_vel_b if self.cfg.obs_ang_vel else None,
            self._robot.data.projected_gravity_b if self.cfg.obs_proj_grav else None,
            (roll.float() + (torch.rand_like(roll) * 0.02 - 0.01)).unsqueeze(-1) if self.cfg.obs_roll else None,
            self.throwing_commands,
            joint_pos_info + (torch.rand_like(joint_pos_info) * 0.02 - 0.01),
            joint_vel_info + (torch.rand_like(joint_vel_info) * 0.1 - 0.05),
            self._actions,
            self.not_released_ball.unsqueeze(-1).float() if self.cfg.obs_notrelease else None,
            noise_displace.unsqueeze(-1) if self.cfg.obs_estimdisplace else None,
            estim_time.unsqueeze(-1) if self.cfg.obs_estimdisplace else None,
            ball_data if self.cfg.obs_estimdisplace else None,
        ]
        obs = torch.cat([t for t in obs_tensors if t is not None], dim=-1)
        obs = torch.clip(obs, -1000.0, 1000.0)

        if torch.sum(torch.isnan(obs)) > 0:
            print("NaNs in observation, per-dim:", torch.sum(torch.isnan(obs), dim=0))

        return {"policy": obs}

    # --------------------------------------------------------------------------
    # Quaternion helpers for vector rotation
    # --------------------------------------------------------------------------
    @staticmethod
    def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by quaternion q (w, x, y, z)."""
        w, x, y, z = q.unbind(dim=-1)
        q_vec = torch.stack((x, y, z), dim=-1)
        t = 2.0 * torch.cross(q_vec, v, dim=-1)
        return v + w.unsqueeze(-1) * t + torch.cross(q_vec, t, dim=-1)

    @staticmethod
    def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by q^{-1} (conjugate quaternion)."""
        w, x, y, z = q.unbind(dim=-1)
        q_conj = torch.stack((w, -x, -y, -z), dim=-1)
        return ThrowingEnv._quat_apply(q_conj, v)

    # --------------------------------------------------------------------------
    # Reward
    # --------------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # ------------------------------------------------------------------
        # Throwing reward: projectile-based distance to target
        # ------------------------------------------------------------------
        dist = torch.norm(
            self.sphere_object.data.body_pos_w[:, 0, 0:3]
            - self._robot.data.body_pos_w[:, [-15], :].squeeze(1),
            dim=1,
        )
        throwing_reward_condition = (dist > 0.25) & (~self.throwing_reward_given)
        self.not_released_ball &= dist <= 0.25
        env_ids = torch.nonzero(throwing_reward_condition).reshape(-1)
        self.throwing_reward[:] = 0.0

        if len(env_ids) > 0:
            self.throwing_reward[env_ids], self.landing_time[env_ids] = self.check_ball_displacement(env_ids)
            self.throwing_reward[env_ids] = 1.0 - self.throwing_reward[env_ids]
            self.throwing_reward_given[env_ids] = True
            self.landing_time[env_ids] += self.episode_length_buf[env_ids] * self.step_dt

        ball_released_envs = env_ids

        # ------------------------------------------------------------------
        # Roll reward (keep torso upright-ish)
        # ------------------------------------------------------------------
        w, x, y, z = self._robot.data.root_quat_w.T
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
        roll = torch.where(torch.isnan(roll), torch.zeros_like(roll), roll)
        roll_rew = (-1.0 / (1.0 + torch.exp(-10.0 * (torch.abs(roll) - 0.3)))) * (
            1.0 - torch.exp(-(torch.abs(roll) - 0.1) / 0.1)
        )

        # ------------------------------------------------------------------
        # Stability reward: base height, ball still in hand, collisions
        # ------------------------------------------------------------------
        base_height = self._robot.data.root_pos_w[:, 2]
        base_height_cond = base_height <= self.min_base_height

        hand_positions = self._robot.data.body_pos_w[:, [-1], :].reshape(self.num_envs, 3)
        ball_positions = self.sphere_object.data.root_pos_w.clone()
        ball_not_thrown_cond = (torch.norm((hand_positions - ball_positions), dim=1) <= 0.25) & (self.reset_buf == 1)

        # collision detection
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)

        # mask out collisions when ball is still in hand
        mask = torch.where(
            torch.norm((hand_positions - ball_positions), dim=1) <= 0.25,
            torch.zeros(1, device=self.device),
            torch.ones(1, device=self.device),
        )
        if self.hand_ids is not None:
            first_contact[:, self.hand_ids] &= mask.view(-1, 1).expand(-1, len(self.hand_ids)).to(torch.bool)

        collision = (
            torch.sum(first_contact[:, self.non_feet_ids], dim=1) > 0
            if self.non_feet_ids is not None
            else torch.zeros_like(base_height, dtype=torch.bool)
        )

        self.stability_penalty_this_ep |= base_height_cond | ball_not_thrown_cond | collision

        if self.cfg.nonsparse_stability_reward:
            safe = ~(base_height_cond | ball_not_thrown_cond | collision)
            stability_rew = (safe.float() / self.cfg.episode_length_s) * self.step_dt
        else:
            stability_rew = ((self.reset_buf == 1) & (~self.stability_penalty_this_ep)).float()

        # ------------------------------------------------------------------
        # New shaping 1: velocity tracking in throw direction
        # ------------------------------------------------------------------
        target_dir_world = self._compute_target_direction_world()
        # end-effector is last body (-1), using linear velocity
        ee_vel_w = self._robot.data.body_vel_w[:, -1, :3]
        ee_speed = torch.norm(ee_vel_w, dim=1, keepdim=True)
        ee_dir = ee_vel_w / (ee_speed + 1e-8)

        cos_align = torch.sum(ee_dir * target_dir_world, dim=1)
        cos_align = torch.clamp(cos_align, min=0.0)  # ignore backwards motion

        vel_tracking_rew = cos_align * self.not_released_ball.float()
        # normalize over episode so sums are ~O(1)
        vel_tracking_rew *= self.step_dt / self.cfg.episode_length_s

        # ------------------------------------------------------------------
        # New shaping 2: fingers-not-blocking (throw dir mostly in EE YZ-plane)
        # ------------------------------------------------------------------
        ee_quat = self._robot.data.body_state_w[:, -1, 3:7]
        # Rotate target_dir into EE frame
        vel_target_ee = self._quat_rotate_inverse(ee_quat, target_dir_world)
        x_component = torch.abs(vel_target_ee[:, 0])  # X-axis in EE frame

        sigma = 0.1
        fingers_not_blocking_rew = 1.0 / (1.0 + x_component / sigma)
        fingers_not_blocking_rew *= self.not_released_ball.float()
        fingers_not_blocking_rew *= self.step_dt / self.cfg.episode_length_s

        # ------------------------------------------------------------------
        # Regularization: action rate, torques, accelerations
        # ------------------------------------------------------------------
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

        rewards = {
            "throwing": torch.max(torch.zeros_like(self.throwing_reward), self.throwing_reward.clone())
            * self.throwing_reward_scale
            * self.cfg.episode_length_s,
            "roll": roll_rew * self.roll_reward_scale * self.step_dt,
            "stability": stability_rew * self.stability_reward_scale * self.cfg.episode_length_s,
            "action_rate_l2": action_rate * self.action_rate_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.joint_accel_reward_scale * self.step_dt,
            "vel_tracking": vel_tracking_rew * self.vel_tracking_reward_scale * self.cfg.episode_length_s,
            "fingers_not_blocking": fingers_not_blocking_rew
            * self.fingers_not_blocking_reward_scale
            * self.cfg.episode_length_s,
        }

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        # Stack into [num_envs, num_reward_components]
        return torch.stack(list(rewards.values())).T

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Time-based termination
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Fall-based termination using calibrated min_base_height
        base_height = self._robot.data.root_pos_w[:, 2]
        if self.min_base_height is not None:
            fall_threshold = self.min_base_height - 0.05
        else:
            # Before calibration, don't terminate by height
            fall_threshold = -1.0
        fallen = base_height < fall_threshold

        died = fallen
        return died, time_out

    # --------------------------------------------------------------------------
    # Reset & curriculum
    # --------------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Calibrate min_base_height once (when a full reset happens)
        if self.min_base_height is None and len(env_ids) == self.num_envs:
            default_root_z = self._robot.data.default_root_state[:, 2]
            self.min_base_height = float(default_root_z.mean().item() - self.fall_margin)
            print(f"[ThrowingEnv] Calibrated min_base_height = {self.min_base_height:.3f}")

        # Body action factor (for curriculum, if used)
        if self.cfg.arm_only:
            self.body_actions = 0.0
        else:
            self.body_actions = 1.0

        if self.cfg.distance_throw:
            self.distance_range = [max(self.distance_range[0], 4.0), max(self.distance_range[1], 4.0)]

        if self.cfg.no_proj_motion:
            self.cfg.obs_estimdisplace = False

        # Reset buffers
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        self._sample_throwing_commands(env_ids)
        self.target_positions[env_ids] = self.calculate_target_offset(env_ids)
        self.throwing_reward[env_ids] = -1.0
        self.landing_time[env_ids] = -1.0
        self.throwing_reward_given[env_ids] = False
        self.not_released_ball[env_ids] = True
        self.sum_open_hand_action[env_ids] = 0.0
        self.stability_penalty_this_ep[env_ids] = False
        self.released_ball_t[env_ids] = -1.0

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, [5, 6, 9, 10, 13, 14, 17, 18, 21, 22]] += torch.zeros_like(
            joint_pos[:, [5, 6, 9, 10, 13, 14, 17, 18, 21, 22]]
        ).uniform_(-0.3, 0.3)
        joint_pos += torch.zeros_like(joint_pos).uniform_(-0.05, 0.05)

        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]

        if self.robot_yaw_offset_rad != 0.0:
            offset_w = math.cos(self.robot_yaw_offset_rad * 0.5)
            offset_z = math.sin(self.robot_yaw_offset_rad * 0.5)
            offset_quat = torch.zeros_like(default_root_state[:, 3:7])
            offset_quat[:, 0] = offset_w
            offset_quat[:, 3] = offset_z
            default_root_state[:, 3:7] = self._quat_multiply(offset_quat, default_root_state[:, 3:7])

        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Cache default root positions for projectile calcs
        self.default_root_states[env_ids] = default_root_state[:, :3]

        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(env_ids)

        # Initialize ball in hand
        object_default_state = self.sphere_object.data.default_root_state.clone()[env_ids]
        finger_positions = self._robot.data.body_pos_w[:, [-1, -4, -5], :][env_ids]
        finger_positions = torch.mean(finger_positions, dim=1).reshape(len(env_ids), 3)
        hand_positions = self._robot.data.body_pos_w[:, [-1], :][env_ids].reshape(len(env_ids), 3)

        vector_AB = finger_positions - hand_positions
        distance_AB = torch.norm(vector_AB, dim=1).reshape(len(env_ids), 1)
        direction = vector_AB / distance_AB
        distance_AC = 0.0 * distance_AB
        position_c = finger_positions  # hand_positions + direction * distance_AC

        object_default_state[:, 0:3] += position_c
        object_default_state[:, 7:] = self._robot.data.body_vel_w[:, [-1], :][env_ids].reshape(len(env_ids), 6)
        self.sphere_object.write_root_state_to_sim(object_default_state, env_ids)

        if self.cfg.no_proj_motion:
            self.ball_distances[env_ids] = -1.0

        self.prev_velocity[env_ids] = torch.zeros((len(env_ids), 3), device=self.device) - 1000.0

        # Logging stats
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            ep_key = "Episode_Reward/" + key
            extras[ep_key] = episodic_sum_avg / self.cfg.episode_length_s
            self._episode_sums[key][env_ids] = 0.0

            if key == "stability":
                extras[ep_key] /= self.stability_reward_scale
            if key == "throwing":
                extras[ep_key] /= self.throwing_reward_scale
            if key == "roll":
                extras[ep_key] /= self.roll_reward_scale
            if key == "vel_tracking":
                extras[ep_key] /= self.vel_tracking_reward_scale
            if key == "fingers_not_blocking":
                extras[ep_key] /= self.fingers_not_blocking_reward_scale

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        term_extras = dict()
        term_extras["Episode_Termination/base_contact"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        term_extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        term_extras["Target_Distance"] = max(self.distance_range)
        term_extras["Theta_Range_Max"] = max(self.theta_range)
        term_extras["Body_Actions"] = self.body_actions
        term_extras["Stability_Timer"] = self.stability_timer
        self.extras["log"].update(term_extras)

    def update_curriculum(self, iter: int):
        """Simple curriculum on target distance / angle."""
        stability_value = self.extras["log"].get("Episode_Reward/stability", 0.0)
        throwing_value = self.extras["log"].get("Episode_Reward/throwing", 0.0)

        if (throwing_value > self.cfg.r_throw_thresh and stability_value > self.cfg.r_stability_thresh) or (
            iter >= 5000
        ):
            if self.cfg.distance_throw:
                self.distance_range = [
                    min(self.cfg.max_throw_dist, self.distance_range[0] + 0.01),
                    min(self.cfg.max_throw_dist, self.distance_range[1] + 0.01),
                ]
            else:
                self.distance_range = [self.distance_range[0], min(self.cfg.max_throw_dist, self.distance_range[1] + 0.01)]
                self.theta_range = [min(0.0, self.theta_range[0]), min(1.0, self.theta_range[1] + 0.01)]

    # --------------------------------------------------------------------------
    # Throw command sampling & geometry utils
    # --------------------------------------------------------------------------
    def _sample_throwing_commands(self, env_ids: torch.Tensor | None):
        """Sample target distance/angles within a forward FOV and height band."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        command_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Azimuth within forward cone
        if self.target_half_fov_rad <= 0.0:
            phi_samples = torch.zeros_like(self.throwing_commands[command_ids, 2])
        else:
            phi_samples = torch.zeros_like(self.throwing_commands[command_ids, 2]).uniform_(
                -self.target_half_fov_rad,
                self.target_half_fov_rad,
            )

        # Sample distance and height, then derive polar elevation (theta)
        dist_min, dist_max = self.distance_range
        z_min, z_max = self.cfg.target_height_range
        margin = 0.05

        z_samples = torch.zeros_like(self.throwing_commands[command_ids, 0]).uniform_(z_min, z_max)
        z_clamped = torch.clamp(z_samples, min=0.0, max=dist_max - margin)

        dist_samples = torch.zeros_like(self.throwing_commands[command_ids, 0]).uniform_(dist_min, dist_max)
        dist_final = torch.max(dist_samples, z_clamped + margin)

        theta_cos = torch.clamp(z_clamped / dist_final, -1.0, 1.0)

        if self.cfg.distance_throw:
            # Keep "flat forward" semantics for distance-throw curriculum
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

    def _compute_target_direction_world(self) -> torch.Tensor:
        """Unit vector in world frame for the nominal throw direction."""
        dist = self.throwing_commands[:, 0]
        theta_cos = torch.clamp(self.throwing_commands[:, 1], -1.0, 1.0)
        phi_body = self.throwing_commands[:, 2]
        theta = torch.arccos(theta_cos)

        x_local = torch.sin(theta) * torch.cos(phi_body)
        y_local = torch.sin(theta) * torch.sin(phi_body)
        z_local = torch.cos(theta)

        yaw = torch.full_like(x_local, self.target_heading_offset_rad)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        x_world = x_local * cos_yaw - y_local * sin_yaw
        y_world = x_local * sin_yaw + y_local * cos_yaw

        dir_world = torch.stack((x_world, y_world, z_local), dim=1)
        dir_world = dir_world / (torch.norm(dir_world, dim=1, keepdim=True) + 1e-8)
        return dir_world

    def calculate_target_offset(self, env_ids: torch.Tensor) -> torch.Tensor:
        command_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        dist = self.throwing_commands[command_ids, 0]
        theta_cos = torch.clamp(self.throwing_commands[command_ids, 1], -1.0, 1.0)
        phi_body = self.throwing_commands[command_ids, 2]
        theta = torch.arccos(theta_cos)

        x_local = dist * torch.sin(theta) * torch.cos(phi_body)
        y_local = dist * torch.sin(theta) * torch.sin(phi_body)
        z_local = dist * torch.cos(theta)

        yaw = torch.full_like(x_local, self.target_heading_offset_rad)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        x_world = x_local * cos_yaw - y_local * sin_yaw
        y_world = x_local * sin_yaw + y_local * cos_yaw

        res = torch.stack((x_world, y_world, z_local), dim=1)
        return res.squeeze(-1)

    # --------------------------------------------------------------------------
    # Projectile-based ball displacement
    # --------------------------------------------------------------------------
    def check_ball_displacement(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ball_data = self.sphere_object.data.body_state_w[env_ids, 0, :10]
        target_positions = self.target_positions[env_ids] + self._terrain.env_origins[env_ids]
        target_positions[:, :2] += self.default_root_states[env_ids, :2]

        if self.cfg.air_resistance:
            raise NotImplementedError("Air resistance case not implemented.")
        else:
            # initial velocities
            vx0 = ball_data[:, 7]
            vy0 = ball_data[:, 8]
            vz0 = ball_data[:, 9]
            a = 9.81

            z0 = ball_data[:, 2]
            z0_minus_0_03 = torch.clamp(z0 - 0.03, min=0.0)
            sqrt_term = torch.sqrt(vz0**2 + 2.0 * a * z0_minus_0_03)

            t1 = (-vz0 + sqrt_term) / -a
            t2 = (-vz0 - sqrt_term) / -a
            tz = torch.max(t1, t2)
            tz = torch.clamp(tz, min=0.0)

            # time grid [steps x batch]
            steps = 100
            frac = torch.linspace(0, 1, steps=steps, device=self.device).unsqueeze(1)
            time_tensor = frac * tz.unsqueeze(0)

            new_x = ball_data[:, 0].unsqueeze(0) + vx0.unsqueeze(0) * time_tensor
            new_y = ball_data[:, 1].unsqueeze(0) + vy0.unsqueeze(0) * time_tensor
            new_z = (
                ball_data[:, 2].unsqueeze(0)
                + vz0.unsqueeze(0) * time_tensor
                - 0.5 * a * time_tensor**2
            )

            distance_command = self.throwing_commands[env_ids, 0].unsqueeze(0)

            disp_matrix = torch.sqrt(
                (new_x - target_positions[:, 0].unsqueeze(0)) ** 2
                + (new_y - target_positions[:, 1].unsqueeze(0)) ** 2
                + (new_z - target_positions[:, 2].unsqueeze(0)) ** 2
            ) / distance_command
            disp_matrix = torch.clamp(disp_matrix, max=1.0)

            disp_min, idx_min = torch.amin(disp_matrix, dim=0), torch.argmin(disp_matrix, dim=0)

            batch_idx = torch.arange(len(env_ids), device=self.device)
            time_at_min = time_tensor[idx_min, batch_idx]

        return disp_min, time_at_min

    # --------------------------------------------------------------------------
    # Target visualization
    # --------------------------------------------------------------------------
    def initialise_target_for_rendering(self, env_ids: torch.Tensor):
        target_default_state = self.target_object.data.default_root_state.clone()[env_ids]
        target_default_state[:, 7:] = torch.zeros_like(self.target_object.data.default_root_state[env_ids, 7:])
        target_default_state[:, 0:3] += self.target_positions[env_ids] + self._terrain.env_origins[env_ids]
        target_default_state[:, :2] += self.default_root_states[env_ids, :2]

        direction_to_target = self._robot.data.default_root_state[env_ids, :3] - self.target_positions[env_ids]
        forward_vector = torch.tensor([[1, 0, 0]], device=target_default_state.device).expand(
            direction_to_target.size(0), -1
        ).float()
        direction_to_target = F.normalize(direction_to_target, p=2, dim=1)

        axis_of_rotation = torch.cross(forward_vector, direction_to_target, dim=1)

        dot_product = (forward_vector * direction_to_target).sum(dim=1, keepdim=True)
        angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
        sin_half_angle = torch.sin(angle / 2.0)

        xyz = axis_of_rotation * sin_half_angle
        w = torch.cos(angle / 2.0)
        quaternion = torch.cat([w, xyz], dim=1)

        current_quaternion = target_default_state[:, 3:7]

        def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
            w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
            w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return torch.stack([w, x, y, z], dim=1)

        new_quaternion = quaternion_multiply(current_quaternion, quaternion)
        target_default_state[:, 3:7] = new_quaternion
        self.target_object.write_root_state_to_sim(target_default_state, env_ids)
