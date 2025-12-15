# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from math import gcd
from typing import Tuple
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

CANONICAL_MAIN_JOINTS = TORSO_JOINTS + RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS
CANONICAL_FINGER_JOINTS = FINGER_JOINTS
CANONICAL_JOINTS = CANONICAL_MAIN_JOINTS + CANONICAL_FINGER_JOINTS

CLOSED_FINGER_TARGETS = (
    0.85, 0.85, 0.85,
    0.65, 0.65, 0.65,
    0.85, 0.85, 0.85,
    0.65, 0.65, 0.65,
)


class ReplicateG1ThrowEnv(DirectRLEnv):
    """
    RECTIFIED VERSION (target-consistent + position-action control + landing-anchored reward)

    Key fixes to make "hit target" actually mean something consistent:
      1) Target position is defined ONCE: target_world = target_positions + env_origins
         (removed any default_root_states XY shifts)
      2) Main joint actions are POSITION DELTAS around default_joint_pos (stable with position targets)
      3) Reward includes REAL landing error once the ball lands (anchors shaping to reality)
    """

    cfg: ReplicateG1ThrowEnvCfg

    def __init__(self, cfg: ReplicateG1ThrowEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # canonical DOF mapping
        self._build_alpha_joint_indices()
        self.main_joint_indices = self._main_joint_ids
        self.finger_joint_indices = self._finger_joint_ids
        self._closed_finger_pose = torch.tensor(CLOSED_FINGER_TARGETS, device=self.device)
        self._q_des = None

        # action buffers
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # commands / targets
        self.throwing_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_positions = torch.zeros(self.num_envs, 3, device=self.device)  # OFFSET in env-local frame

        # misc buffers
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

        self.stability_penalty_this_ep = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.curriculum = True
        self.action_noise = 0.0
        self.prev_velocity = torch.zeros(self.num_envs, 3, device=self.device) - 1000

        # for optional safety termination
        self._joint_vel_violation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # reward logging
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

        # config-dependent ranges
        if self.cfg.arm_only:
            self.body_actions = 0.0
        else:
            self.body_actions = 1.0

        if self.cfg.distance_throw:
            self.distance_range = [4, 4]
            self.theta_range = [1.0, 1.0]
        else:
            init_max_dist = getattr(self.cfg, "initial_max_throw_dist", self.cfg.max_throw_dist)
            init_max_dist = min(self.cfg.max_throw_dist, max(self.cfg.min_throw_dist, init_max_dist))
            self.distance_range = [self.cfg.min_throw_dist, init_max_dist]
            self.theta_range = [0.0, 1.0]

        init_hmin, init_hmax = getattr(self.cfg, "initial_target_height_range", self.cfg.target_height_range)
        hmin = max(self.cfg.target_height_range[0], init_hmin)
        hmax = min(self.cfg.target_height_range[1], init_hmax)
        if hmax < hmin:
            hmax = hmin
        self.current_target_height_range = [hmin, hmax]

        self.target_half_fov_rad = min(math.radians(self.cfg.target_fov_deg) / 2.0, math.pi)
        self.target_heading_offset_rad = math.radians(self.cfg.target_heading_offset_deg)
        self.robot_yaw_offset_rad = math.radians(self.cfg.robot_yaw_offset_deg)

        if self.cfg.no_proj_motion:
            self.cfg.obs_estimdisplace = False

        self._sample_throwing_commands(torch.arange(self.num_envs, device=self.device))

        # internal desired joint targets (for position-action semantics)
        self._q_des = self._robot.data.joint_pos.clone() if hasattr(self, "_robot") else None

    # ---------------------------------------------------------------------
    # Scene
    # ---------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.sphere_object = RigidObject(self.cfg.sphere_cfg)
        self.scene.rigid_objects["sphere"] = self.sphere_object

        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.target_object = RigidObject(self.cfg.target_cfg)
            self.scene.rigid_objects["target"] = self.target_object

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self._apply_env_color_pairs()
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # lights
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        light_cfg.func("/World/Light", light_cfg, orientation=(1.0, 0.0, 0.0, 0.0))

    # ---------------------------------------------------------------------
    # Joint index mapping
    # ---------------------------------------------------------------------
    def _build_alpha_joint_indices(self):
        if (
            hasattr(self, "_main_joint_ids")
            and hasattr(self, "_finger_joint_ids")
            and hasattr(self, "_right_arm_ids")
            and hasattr(self, "_other_main_joint_ids")
        ):
            return

        robot_joint_names = list(self._robot.data.joint_names)
        name_to_idx = {name: i for i, name in enumerate(robot_joint_names)}

        missing = [j for j in CANONICAL_JOINTS if j not in name_to_idx]
        assert not missing, f"Missing joints in robot model: {missing}"

        main_indices = [name_to_idx[j] for j in CANONICAL_MAIN_JOINTS]
        finger_indices = [name_to_idx[j] for j in CANONICAL_FINGER_JOINTS]
        right_arm_indices = [name_to_idx[j] for j in RIGHT_ARM_JOINTS]
        other_main_indices = [idx for idx in main_indices if idx not in right_arm_indices]

        self._main_joint_ids = torch.tensor(main_indices, device=self.device, dtype=torch.long)
        self._finger_joint_ids = torch.tensor(finger_indices, device=self.device, dtype=torch.long)
        self._right_arm_ids = torch.tensor(right_arm_indices, device=self.device, dtype=torch.long)
        self._other_main_joint_ids = torch.tensor(other_main_indices, device=self.device, dtype=torch.long)
        self._non_finger_joint_ids = self._main_joint_ids

        assert len(self._main_joint_ids) == 15
        assert len(self._finger_joint_ids) == 12
        assert len(self._right_arm_ids) == len(RIGHT_ARM_JOINTS)

    # ---------------------------------------------------------------------
    # Target WORLD position helper (single source of truth)
    # ---------------------------------------------------------------------
    def _target_world(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            return self.target_positions + self._terrain.env_origins
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        return self.target_positions[env_ids] + self._terrain.env_origins[env_ids]

    # ---------------------------------------------------------------------
    # Action processing (FIX: position-delta around default, not velocity integration)
    # ---------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._actions = actions

        self._build_alpha_joint_indices()
        main_idx = self._main_joint_ids
        finger_idx = self._finger_joint_ids

        num_main = main_idx.shape[0]      # 15
        num_fingers = finger_idx.shape[0] # 12
        num_dofs = self._robot.data.joint_pos.shape[1]
        assert num_main + num_fingers == num_dofs, f"DOF mismatch: main={num_main}, fingers={num_fingers}, total={num_dofs}"

        q0 = self._robot.data.default_joint_pos
        self._processed_actions = self._robot.data.joint_pos.clone()

        # MAIN: action is delta-position (rad) around default
        # action_scale should be in "radians of range" (e.g., 0.3, 0.6, 1.0 etc.)
        delta_q = self.cfg.action_scale * self._actions[:, :num_main]
        self._processed_actions[:, main_idx] = q0[:, main_idx] + delta_q

        # FINGERS: grip scalar open/close
        grip_action = self._actions[:, num_main]  # (N,)
        closed = self._closed_finger_pose.unsqueeze(0).expand(self.num_envs, -1)
        open_ = torch.zeros_like(closed)
        hand_close_mask = (grip_action <= 0).view(-1, 1)
        finger_targets = torch.where(hand_close_mask, closed, open_)
        self._processed_actions[:, finger_idx] = finger_targets

        # clip to limits if available
        limits = getattr(self._robot.data, "joint_pos_limits", None)
        if limits is not None:
            lower, upper = limits[:, :, 0], limits[:, :, 1]
            self._processed_actions = torch.clamp(self._processed_actions, lower, upper)

        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(torch.arange(self.num_envs, device=self.device))

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._processed_actions)

    # ---------------------------------------------------------------------
    # Observations (kept mostly as-is, but target is consistent via _target_world())
    # ---------------------------------------------------------------------
    def _get_observations(self) -> dict:
        self._build_alpha_joint_indices()

        if not hasattr(self, "non_feet_ids"):
            self.non_feet_ids, _ = self._contact_sensor.find_bodies("^(?!.*ankle).*$")
            self.hand_ids, _ = self._contact_sensor.find_bodies(".*(palm|five|six|three|four|zero|one|two).*")
            self.left_hand_ids, _ = self._contact_sensor.find_bodies(".*left_(palm).*")
            self.right_hand_ids, _ = self._contact_sensor.find_bodies(".*right_(palm).*")

        idx = self._non_finger_joint_ids
        joint_pos_info = self._robot.data.joint_pos[:, idx] - self._robot.data.default_joint_pos[:, idx]
        joint_vel_info = self._robot.data.joint_vel[:, idx]
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

        estimated_displacement, estim_time = self.check_ball_displacement(torch.arange(self.num_envs, device=self.device))
        estimated_displacement = 1 - estimated_displacement

        w, x, y, z = self._robot.data.root_quat_w.T
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2)).to(self.device)

        noise_displace = (1 - self.action_noise) * estimated_displacement + self.action_noise * torch.randn_like(estimated_displacement)
        self.prev_velocity = self._robot.data.root_lin_vel_b.clone()

        ball_data = self.sphere_object.data.body_state_w[:, 0, [7, 8, 9]]  # vx,vy,vz

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_ang_vel_b if self.cfg.obs_ang_vel else None,
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
        obs = torch.clip(obs, -1000, 1000)
        return {"policy": obs}

    # ---------------------------------------------------------------------
    # Rewards (FIX: adds landing-anchored reward, keeps your dense shaping)
    # ---------------------------------------------------------------------
    def _ensure_throw_hand_ids(self):
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
            assert len(palm_candidates) > 0, "No palm body found"
            self._throw_hand_body_id = palm_candidates[-1]

        if hand_side == "right":
            fingertip_ids, _ = self._contact_sensor.find_bodies(".*right_(thumb|index|pinky)_distal.*")
        else:
            fingertip_ids, _ = self._contact_sensor.find_bodies(".*left_(thumb|index|pinky)_distal.*")
        if len(fingertip_ids) == 0:
            fingertip_ids, _ = self._contact_sensor.find_bodies(".*(thumb|index|pinky).*distal.*")
        self._right_fingertip_ids = fingertip_ids

    def _get_rewards(self) -> torch.Tensor:
        self._ensure_throw_hand_ids()
        self._build_alpha_joint_indices()
        idx = self._non_finger_joint_ids

        ball_pos = self.sphere_object.data.body_pos_w[:, 0, 0:3]
        hand_pos = self._robot.data.body_pos_w[:, self._throw_hand_body_id, :]
        dist_hand_ball = torch.norm(ball_pos - hand_pos, dim=1)
        release_threshold = float(getattr(self.cfg, "release_distance_threshold", 0.25))

        # release event
        release_mask = (dist_hand_ball > release_threshold) & (~self.throwing_reward_given)
        self.not_released_ball &= ~release_mask
        ball_released_envs = torch.nonzero(release_mask, as_tuple=False).flatten()

        self.throwing_reward.zero_()
        projectile_rew = torch.zeros(self.num_envs, device=self.device)

        if ball_released_envs.numel() > 0:
            disp, landing_time = self.check_ball_displacement(ball_released_envs)
            # projectile reward: r_throw = 1 - min(E/r, 1) where E is displacement to target
            proj_rew_vals = torch.clamp(1.0 - disp, min=0.0)
            self.throwing_reward[ball_released_envs] = proj_rew_vals
            projectile_rew[ball_released_envs] = proj_rew_vals
            self.throwing_reward_given[ball_released_envs] = True
            self.landing_time[ball_released_envs] = landing_time + self.episode_length_buf[ball_released_envs] * self.step_dt
            self.release_ball_pos[ball_released_envs] = ball_pos[ball_released_envs]

            target_pos = self._target_world(ball_released_envs)
            self.release_target_dir[ball_released_envs] = target_pos - ball_pos[ball_released_envs]

        # stability / collision penalty (kept)
        ball_positions = self.sphere_object.data.root_pos_w.clone()
        ball_not_thrown_cond = (torch.norm((hand_pos - ball_positions), dim=1) <= release_threshold) & (self.reset_buf == 1)

        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)
        mask = torch.where(
            torch.norm((hand_pos - ball_positions), dim=1) <= release_threshold,
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

        # smoothness penalties
        action_rate = torch.mean(torch.square(self._actions[:, : idx.shape[0]] - self._previous_actions[:, : idx.shape[0]]), dim=1)
        joint_torques = torch.mean(torch.square(self._robot.data.applied_torque[:, idx]), dim=1)
        # use sim-reported joint accelerations if available, otherwise fallback to zeros
        joint_acc_data = getattr(self._robot.data, "joint_acc", None)
        if joint_acc_data is not None:
            joint_accel = torch.mean(torch.square(joint_acc_data[:, idx]), dim=1)
        else:
            joint_accel = torch.zeros_like(action_rate)

        # ball release bonus (single step)
        ballrel_rew = torch.zeros(self.num_envs, device=self.device)
        if ball_released_envs.numel() > 0:
            ballrel_rew[ball_released_envs] = getattr(self.cfg, "ball_release_reward_scale", 0.0)

        # overall rewards
        rewards = {
            "throwing": torch.clamp(self.throwing_reward, min=0.0) * self.cfg.throwing_reward_scale,
            "projectile_rew": projectile_rew * float(getattr(self.cfg, "projectile_reward_scale", 0.0)),
            "stability": stability_rew * self.cfg.stability_reward_scale,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "ballrel_rew": ballrel_rew,
        }

        # debug logs
        if not hasattr(self, "extras"):
            self.extras = {}
        self.extras["log"] = self.extras.get("log", {})
        n_env = float(self.num_envs)
        self.extras["log"]["debug/collision_pct"] = 100.0 * collision.float().mean()
        self.extras["log"]["debug/ball_not_thrown_pct"] = 100.0 * ball_not_thrown_cond.float().mean()
        self.extras["log"]["debug/stability_penalty_pct"] = 100.0 * self.stability_penalty_this_ep.float().mean()
        # per-step: only counts envs releasing this step (expected small)
        self.extras["log"]["debug/ball_released_pct"] = 100.0 * (ball_released_envs.numel() / max(n_env, 1.0))
        # cumulative: fraction of envs that have released at least once this episode
        released_any = (~self.not_released_ball).float().mean() * 100.0
        self.extras["log"]["debug/ball_released_cum_pct"] = released_any

        # curriculum update hook
        try:
            self.update_curriculum(self.common_step_counter)
        except Exception:
            pass

        for k, v in rewards.items():
            self._episode_sums[k] += v

        self._previous_actions = self._actions.clone()

        reward_vec = torch.stack([rewards[name] for name in self.reward_component_names], dim=1)
        return reward_vec

    # ---------------------------------------------------------------------
    # Dones (FIX: uses the same target_world definition)
    # ---------------------------------------------------------------------
    def _check_joint_velocity_violation(self) -> tuple[torch.Tensor, torch.Tensor]:
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

        ball_pos = self.sphere_object.data.root_pos_w  # (N,3)
        target_pos = self._target_world()              # (N,3)

        released = ~self.not_released_ball
        hit_ground = (ball_pos[:, 2] < 0.05) & released
        too_far = released & (torch.norm(ball_pos[:, :2] - target_pos[:, :2], dim=1) > self.cfg.max_throw_dist + 1.0)

        joint_vel_violation, _ = self._check_joint_velocity_violation()

        timeout_frac = getattr(self.cfg, "no_release_timeout_frac", 0.75)
        no_release_timeout = (self.episode_length_buf > timeout_frac * self.max_episode_length) & (~released)

        terminated = hit_ground | too_far | no_release_timeout | joint_vel_violation

        success_radius = float(getattr(self.cfg, "success_radius", 0.3))
        success = hit_ground & (torch.norm(ball_pos - target_pos, dim=1) < success_radius)

        self.extras["termination"] = {
            "terminated": terminated,
            "truncated": time_out,
            "success": success,
            "joint_vel_violation": joint_vel_violation,
        }
        return terminated, time_out

    # ---------------------------------------------------------------------
    # Reset (FIX: still resets ball into hand; target consistent)
    # ---------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._build_alpha_joint_indices()
        grip_action_index = self._main_joint_ids.shape[0]

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._actions[env_ids, grip_action_index] = -1.0
        self._previous_actions[env_ids, grip_action_index] = -1.0

        for k in self._episode_sums.keys():
            self._episode_sums[k][env_ids] = 0.0

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

        # sample commands + compute target offsets
        self._sample_throwing_commands(env_ids)
        self.target_positions[env_ids] = self.calculate_target_offset(env_ids)

        # joint reset
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self._robot.data.default_joint_vel[env_ids])

        # fingers start closed to hold ball
        joint_pos[:, self._finger_joint_ids] = self._closed_finger_pose

        # randomized initialization
        right_arm_range = float(getattr(self.cfg, "right_arm_init_range", 0.0))
        if right_arm_range > 0.0 and self._right_arm_ids.numel() > 0:
            right_arm_noise = torch.zeros_like(joint_pos[:, self._right_arm_ids])
            right_arm_noise.uniform_(-right_arm_range, right_arm_range)
            joint_pos[:, self._right_arm_ids] += right_arm_noise

        other_joint_range = float(getattr(self.cfg, "other_joint_init_range", 0.0))
        other_joint_ids = torch.cat((self._other_main_joint_ids, self._finger_joint_ids))
        if other_joint_range > 0.0 and other_joint_ids.numel() > 0:
            other_noise = torch.zeros_like(joint_pos[:, other_joint_ids])
            other_noise.uniform_(-other_joint_range, other_joint_range)
            joint_pos[:, other_joint_ids] += other_noise

        pos_noise_range = getattr(self.cfg, "joint_pos_noise_range", None)
        if pos_noise_range is not None:
            joint_pos += torch.zeros_like(joint_pos).uniform_(pos_noise_range[0], pos_noise_range[1])

        vel_noise_range = getattr(self.cfg, "joint_vel_noise_range", None)
        if vel_noise_range is not None:
            joint_vel += torch.zeros_like(joint_vel).uniform_(vel_noise_range[0], vel_noise_range[1])

        limits = getattr(self._robot.data, "joint_pos_limits", None)
        if limits is not None:
            lower = limits[env_ids, :, 0]
            upper = limits[env_ids, :, 1]
            joint_pos = torch.max(torch.min(joint_pos, upper), lower)

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 7:] = 0.0

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
        self._q_des = self._robot.data.joint_pos.clone()

        self._reset_ball_to_hand(env_ids)

        # render target
        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(env_ids)

    def _reset_ball_to_hand(self, env_ids: Sequence[int] | torch.Tensor):
        if env_ids is None or len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._ensure_throw_hand_ids()

        ball_state = self.sphere_object.data.default_root_state.clone()[env_ids]
        hand_pos = self._robot.data.body_pos_w[env_ids, self._throw_hand_body_id]
        hand_quat = self._robot.data.body_quat_w[env_ids, self._throw_hand_body_id]

        forward_offset_local = torch.tensor([0.0, 0.08, 0.04], device=self.device)
        forward_offset_world = self._quat_apply(hand_quat, forward_offset_local.expand(len(env_ids), -1))
        ball_offset = forward_offset_world

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

        ball_pos = hand_pos + ball_offset
        ball_state[:, :3] = ball_pos
        ball_state[:, 3:7] = self.sphere_object.data.default_root_state[env_ids, 3:7]
        ball_state[:, 7:] = 0.0
        self.sphere_object.write_root_state_to_sim(ball_state, env_ids)

    # ---------------------------------------------------------------------
    # Curriculum (unchanged)
    # ---------------------------------------------------------------------
    def update_curriculum(self, iter):
        log_dict = self.extras.get("log", {}) if hasattr(self, "extras") else {}
        stability_value = log_dict.get("Episode_Reward/stability", None)
        throwing_value = log_dict.get("Episode_Reward/throwing", None)
        if stability_value is None or throwing_value is None:
            return

        self.action_noise = min(1.0, self.action_noise + 0.001)

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
                self.theta_range = [min(0.0, self.theta_range[0]), min(1.0, self.theta_range[1] + 0.01)]

            new_height_max = min(self.cfg.target_height_range[1], self.current_target_height_range[1] + height_step)
            self.current_target_height_range[1] = new_height_max

    # ---------------------------------------------------------------------
    # Command sampling + target offset
    # ---------------------------------------------------------------------
    def _sample_throwing_commands(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        command_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

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

        if fixed_dist is not None:
            dist_final = torch.full_like(self.throwing_commands[command_ids, 0], float(fixed_dist))
        else:
            dist_min, dist_max = self.distance_range
            dist_final = torch.zeros_like(self.throwing_commands[command_ids, 0]).uniform_(dist_min, dist_max)

        if fixed_height is not None:
            z_clamped = torch.full_like(self.throwing_commands[command_ids, 0], float(fixed_height))
        else:
            z_min, z_max = self.current_target_height_range
            z_samples = torch.zeros_like(self.throwing_commands[command_ids, 0]).uniform_(z_min, z_max)
            z_clamped = torch.clamp(z_samples, min=0.0, max=dist_final.max() - 0.05)

        theta_cos = torch.clamp(z_clamped / torch.clamp(dist_final, min=1e-3), -1.0, 1.0)

        if self.cfg.distance_throw:
            theta_cos = torch.ones_like(theta_cos)
            phi_samples = torch.zeros_like(phi_samples)

        self.throwing_commands[command_ids, 0] = dist_final
        self.throwing_commands[command_ids, 1] = theta_cos
        self.throwing_commands[command_ids, 2] = phi_samples

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
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

        x_world = x_local * cos_yaw - y_local * sin_yaw
        y_world = x_local * sin_yaw + y_local * cos_yaw
        return torch.stack((x_world, y_world, z_local), dim=1)

    def check_ball_displacement(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ball_data = self.sphere_object.data.body_state_w[env_ids, 0, :10]
        target_positions = self._target_world(env_ids)  # FIX: single source of truth

        if self.cfg.air_resistance:
            raise NotImplementedError
        else:
            vx0, vy0, vz0 = ball_data[:, 7], ball_data[:, 8], ball_data[:, 9]
            a = 9.81

            z0 = ball_data[:, 2]
            z0_minus_0_03 = torch.clamp(z0 - 0.03, min=0.0)
            sqrt_term = torch.sqrt(vz0**2 + 2 * a * z0_minus_0_03)

            t1 = (-vz0 + sqrt_term) / -a
            t2 = (-vz0 - sqrt_term) / -a
            tz = torch.max(t1, t2)
            tz = torch.clamp(tz, min=0.0)

            steps = 100
            frac = torch.linspace(0, 1, steps=steps, device=self.device).unsqueeze(1)
            time_tensor = frac * tz.unsqueeze(0)

            new_x = ball_data[:, 0].unsqueeze(0) + vx0.unsqueeze(0) * time_tensor
            new_y = ball_data[:, 1].unsqueeze(0) + vy0.unsqueeze(0) * time_tensor
            new_z = ball_data[:, 2].unsqueeze(0) + vz0.unsqueeze(0) * time_tensor - 0.5 * a * time_tensor**2

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
    # Rendering target (FIX: uses target_world only; no extra offsets)
    # ---------------------------------------------------------------------
    def initialise_target_for_rendering(self, env_ids):
        target_default_state = self.target_object.data.default_root_state.clone()[env_ids]
        target_default_state[:, 7:] = 0.0

        target_default_state[:, 0:3] = self._target_world(env_ids)  # FIX: consistent placement

        # Fix orientation: rotate 90 deg about Z so the board normal points +Y
        half_angle = math.pi / 4.0
        target_default_state[:, 3:7] = torch.tensor(
            [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)],
            device=self.device,
        )

        self.target_object.write_root_state_to_sim(target_default_state, env_ids)

    # ---------------------------------------------------------------------
    # Misc: yaw / quats / colors (your originals, unchanged)
    # ---------------------------------------------------------------------
    def _get_base_yaw(self, env_ids: torch.Tensor) -> torch.Tensor:
        command_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        base_quats = self._robot.data.root_quat_w[command_ids]
        w, x, y, z = base_quats.unbind(dim=1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.where(torch.isnan(yaw), torch.zeros_like(yaw), yaw)

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(dim=1)
        w2, x2, y2, z2 = q2.unbind(dim=1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack((w, x, y, z), dim=1)

    @staticmethod
    def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
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
        if num_envs <= 1:
            return list(range(num_envs))
        step = max(1, num_envs // 3 or 1)
        while gcd(step, num_envs) != 1:
            step += 1
        order = []
        idx = 0
        for _ in range(num_envs):
            order.append(idx)
            idx = (idx + step) % num_envs
        return order
