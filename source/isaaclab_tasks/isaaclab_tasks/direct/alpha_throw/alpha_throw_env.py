# alpha_throw_env.py

from __future__ import annotations

import math
from typing import Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F
import isaacsim.core.utils.stage as stage_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .alpha_throw_env_cfg import AlphaThrowEnvCfg


class AlphaThrowEnv(DirectRLEnv):
    cfg: AlphaThrowEnvCfg

    def __init__(self, cfg: AlphaThrowEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action storage
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions = torch.zeros_like(self._actions)

        # Throwing command: [distance, cos(theta), phi]
        self.throwing_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_positions = torch.zeros(self.num_envs, 3, device=self.device)

        # Flags and reward buffers
        self.not_released_ball = torch.ones(self.num_envs, device=self.device).bool()
        self.throwing_reward = torch.ones(self.num_envs, device=self.device) * -1.0
        self.throwing_reward_given = torch.zeros(self.num_envs, device=self.device).bool()
        self.landing_time = torch.ones(self.num_envs, device=self.device) * -1.0

        # Collision buffer (for termination)
        self.collision_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Logging (episodic sums)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "throwing",
                "action_rate_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "collision_penalty",
            ]
        }
        self.reward_component_names = list(self._episode_sums.keys())
        self.reward_components = len(self.reward_component_names)

        # Distance / target sampling ranges
        self.distance_range = [4.0, 8.0]
        self.theta_range = [0.0, 1.0]
        self.target_half_fov_rad = min(math.radians(self.cfg.target_fov_deg) / 2.0, math.pi)
        self.target_heading_offset_rad = math.radians(self.cfg.target_heading_offset_deg)
        self.robot_yaw_offset_rad = math.radians(self.cfg.robot_yaw_offset_deg)
        self._sample_throwing_commands(torch.arange(self.num_envs, device=self.device))

        # For air-resistance free check_ball_displacement
        self.default_root_states = torch.zeros(self.num_envs, 3, device=self.device)
        self.released_ball_t = torch.zeros(self.num_envs, device=self.device) - 1.0

        self.action_noise = 0.0

        # Useful caches for contact sensor body ids
        self.hand_ids = None
        self.arm_body_ids = None  # bodies we consider for self-collision

    # ------------------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------------------

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

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone, filter, replicate
        self.scene.clone_environments(copy_from_source=False)
        self._apply_env_color_pairs()

        # Filter collisions against the ground only
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Light
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}"
                "/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
            ),
        )
        light_cfg.func("/World/Light", light_cfg, orientation=(1.0, 0.0, 0.0, 0.0))

    # ------------------------------------------------------------------------------
    # Color assignment (ball/target per env, same as your G1 env)
    # ------------------------------------------------------------------------------

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

    def _generate_env_color_palette(self, num_envs: int):
        if num_envs <= 0:
            return []
        saturation, value = 0.75, 0.9
        colors = []
        for env_index in range(num_envs):
            hue = (env_index / max(num_envs, 1)) % 1.0
            colors.append(self._hsv_to_rgb(hue, saturation, value))
        return colors

    @staticmethod
    def _set_shader_color(stage, shader_path: str, color):
        prim = stage.GetPrimAtPath(shader_path)
        if prim.IsValid():
            sim_utils.safe_set_attribute_on_usd_prim(
                prim, "inputs:diffuseColor", color, camel_case=False
            )

    @staticmethod
    def _hsv_to_rgb(hue: float, saturation: float, value: float):
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

    def _color_distribution_indices(self, num_envs: int):
        from math import gcd

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

    # ------------------------------------------------------------------------------
    # RL hooks
    # ------------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._previous_actions = self._actions.clone()

        default_positions = self._robot.data.default_joint_pos.clone()  # [num_envs, 15]
        self._processed_actions = self.cfg.action_scale * self._actions + default_positions

        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(torch.arange(self.num_envs, device=self.device))

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        # Cache body ids for collision handling
        if self.hand_ids is None or self.arm_body_ids is None:
            # hands: RHAND, LHAND
            self.hand_ids, _ = self._contact_sensor.find_bodies(".*(RHAND|LHAND).*")
            # all arm bodies (RLx, LLx and hands)
            self.arm_body_ids, _ = self._contact_sensor.find_bodies(".*(RL[0-9]|LL[0-9]|RHAND|LHAND).*")

        joint_pos_info = (self._robot.data.joint_pos - self._robot.data.default_joint_pos)
        joint_vel_info = self._robot.data.joint_vel

        # Ball displacement estimate (same API as your original env)
        estimated_displacement, estim_time = self.check_ball_displacement(
            torch.arange(self.num_envs, device=self.device)
        )
        estimated_displacement = 1.0 - estimated_displacement  # closer → higher value

        # Ball linear velocity
        ball_vel = self.sphere_object.data.body_state_w[:, 0, 7:10]

        # Assemble obs
        obs_tensors = []

        if self.cfg.obs_ang_vel:
            obs_tensors.append(self._robot.data.root_ang_vel_b)

        if self.cfg.obs_lin_vel:
            obs_tensors.append(self._robot.data.root_lin_vel_b)

        if self.cfg.obs_proj_grav:
            obs_tensors.append(self._robot.data.projected_gravity_b)

        # throwing command [dist, cos(theta), phi]
        obs_tensors.append(self.throwing_commands)

        # joint pos/vel with small noise
        obs_tensors.append(
            joint_pos_info + (torch.rand_like(joint_pos_info) * 0.02 - 0.01)
        )
        obs_tensors.append(
            joint_vel_info + (torch.rand_like(joint_vel_info) * 0.1 - 0.05)
        )

        # last action
        obs_tensors.append(self._actions)

        if self.cfg.obs_notrelease:
            obs_tensors.append(self.not_released_ball.unsqueeze(-1).float())

        if self.cfg.obs_estimdisplace:
            obs_tensors.append(estimated_displacement.unsqueeze(-1))
            obs_tensors.append(estim_time.unsqueeze(-1))
            obs_tensors.append(ball_vel)

        obs = torch.cat(obs_tensors, dim=-1)
        obs = torch.clip(obs, -1000.0, 1000.0)

        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # --------------------------------------------------------------
        # 1) Detect ball release and compute throwing reward
        # --------------------------------------------------------------
        # distance ball–right hand
        # we assume RHAND is in arm_body_ids; better would be to explicitly find its index
        if not hasattr(self, "right_hand_idx"):
            body_ids, _ = self._contact_sensor.find_bodies(".*RHAND.*")
            assert len(body_ids) > 0, "Could not find RHAND in contact sensor bodies"
            self.right_hand_idx = body_ids[0]

        ball_pos = self.sphere_object.data.body_pos_w[:, 0, 0:3]
        hand_pos = self._robot.data.body_pos_w[:, self.right_hand_idx, :3]
        dist = torch.norm(ball_pos - hand_pos, dim=1)

        throwing_reward_condition = (dist > 0.25) & (~self.throwing_reward_given)
        self.not_released_ball &= (dist <= 0.25)

        env_ids = torch.nonzero(throwing_reward_condition).reshape(-1)
        self.throwing_reward[:] = 0.0
        if len(env_ids) > 0:
            self.throwing_reward[env_ids], self.landing_time[env_ids] = self.check_ball_displacement(env_ids)
            self.throwing_reward[env_ids] = 1.0 - self.throwing_reward[env_ids]
            self.throwing_reward_given[env_ids] = True
            self.landing_time[env_ids] += self.episode_length_buf[env_ids] * self.step_dt

        # clamp to non-negative
        throwing_term = torch.clamp(self.throwing_reward.clone(), min=0.0)

        # --------------------------------------------------------------
        # 2) Self-collision detection & penalty
        # --------------------------------------------------------------
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)

        # Ignore hand–ball contacts based on distance threshold
        mask_good_contact = (dist <= 0.25).view(-1, 1).expand(-1, len(self.hand_ids))
        first_contact[:, self.hand_ids] &= ~mask_good_contact.to(torch.bool)

        collision = torch.sum(first_contact[:, self.arm_body_ids], dim=1) > 0
        self.collision_buf |= collision

        collision_penalty = -self.cfg.collision_penalty_scale * collision.float()

        # --------------------------------------------------------------
        # 3) Regularization terms
        # --------------------------------------------------------------
        action_rate = torch.sum((self._actions - self._previous_actions) ** 2, dim=1)
        joint_torques = torch.sum(self._robot.data.applied_torque ** 2, dim=1)
        joint_accel = torch.sum(self._robot.data.joint_acc ** 2, dim=1)

        rewards = {
            "throwing": throwing_term * self.cfg.throwing_reward_scale * self.max_episode_length_s,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "collision_penalty": collision_penalty,
        }

        # Log episodic sums
        for key, value in rewards.items():
            self._episode_sums[key] += value

        # Return stacked reward vector [num_envs, num_components]
        return torch.stack(list(rewards.values()), dim=-1)

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.collision_buf.clone()
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.throwing_reward[env_ids] = -1.0
        self.landing_time[env_ids] = -1.0
        self.throwing_reward_given[env_ids] = False
        self.not_released_ball[env_ids] = True
        self.released_ball_t[env_ids] = -1.0
        self.collision_buf[env_ids] = False

        # Sample new throwing commands and target offsets
        self._sample_throwing_commands(env_ids)
        self.target_positions[env_ids] = self.calculate_target_offset(env_ids)

        # Reset robot joint state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
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

        # Initialize ball "in hand"
        # We place it at RHAND position (can refine later with an offset)
        if not hasattr(self, "right_hand_idx"):
            body_ids, _ = self._contact_sensor.find_bodies(".*RHAND.*")
            assert len(body_ids) > 0, "Could not find RHAND in contact sensor bodies"
            self.right_hand_idx = body_ids[0]

        hand_positions = self._robot.data.body_pos_w[:, self.right_hand_idx, :3][env_ids]  # [Ne, 3]
        object_default_state = self.sphere_object.data.default_root_state[env_ids].clone()
        object_default_state[:, 0:3] = hand_positions
        object_default_state[:, 7:] = 0.0  # no initial velocity (or copy hand vel if you want)
        self.sphere_object.write_root_state_to_sim(object_default_state, env_ids)

        # Render target
        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        self.extras["log"]["Target_Distance_Max"] = max(self.distance_range)

    # ------------------------------------------------------------------------------
    # Curriculum / command sampling
    # ------------------------------------------------------------------------------

    def _sample_throwing_commands(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        command_ids = env_ids.long()

        # Azimuth within FOV
        if self.target_half_fov_rad <= 0.0:
            phi_samples = torch.zeros_like(self.throwing_commands[command_ids, 2])
        else:
            phi_samples = torch.zeros_like(self.throwing_commands[command_ids, 2]).uniform_(
                -self.target_half_fov_rad, self.target_half_fov_rad
            )

        dist_min, dist_max = self.distance_range
        z_min, z_max = self.cfg.target_height_range
        margin = 0.05

        z_samples = torch.zeros_like(self.throwing_commands[command_ids, 0]).uniform_(z_min, z_max)
        z_clamped = torch.clamp(z_samples, min=0.0, max=dist_max - margin)

        dist_samples = torch.zeros_like(self.throwing_commands[command_ids, 0]).uniform_(dist_min, dist_max)
        dist_final = torch.max(dist_samples, z_clamped + margin)

        theta_cos = torch.clamp(z_clamped / dist_final, -1.0, 1.0)

        self.throwing_commands[command_ids, 0] = dist_final
        self.throwing_commands[command_ids, 1] = theta_cos
        self.throwing_commands[command_ids, 2] = phi_samples

    def calculate_target_offset(self, env_ids: torch.Tensor) -> torch.Tensor:
        command_ids = env_ids.long()
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
        return res

    # ------------------------------------------------------------------------------
    # Ball displacement check (no air resistance by default)
    # ------------------------------------------------------------------------------

    def check_ball_displacement(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ball_data = self.sphere_object.data.body_state_w[env_ids, 0, :10]
        target_positions = self.target_positions[env_ids] + self._terrain.env_origins[env_ids]

        if self.cfg.air_resistance:
            raise NotImplementedError("Air resistance model not implemented yet.")
        else:
            vx0 = ball_data[:, 7]
            vy0 = ball_data[:, 8]
            vz0 = ball_data[:, 9]
            a = 9.81

            z0 = ball_data[:, 2]
            z0_minus_0_03 = torch.clamp(z0 - 0.03, min=0.0)
            sqrt_term = torch.sqrt(vz0 ** 2 + 2 * a * z0_minus_0_03)

            t1 = (-vz0 + sqrt_term) / -a
            t2 = (-vz0 - sqrt_term) / -a
            tz = torch.max(t1, t2)
            tz = torch.clamp(tz, min=0.0)

            steps = 100
            frac = torch.linspace(0, 1, steps=steps, device=self.device).unsqueeze(1)
            time_tensor = frac * tz.unsqueeze(0)

            new_x = ball_data[:, 0].unsqueeze(0) + vx0.unsqueeze(0) * time_tensor
            new_y = ball_data[:, 1].unsqueeze(0) + vy0.unsqueeze(0) * time_tensor
            new_z = (
                ball_data[:, 2].unsqueeze(0)
                + vz0.unsqueeze(0) * time_tensor
                - 0.5 * a * time_tensor ** 2
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

    # ------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(dim=1)
        w2, x2, y2, z2 = q2.unbind(dim=1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack((w, x, y, z), dim=1)

    def initialise_target_for_rendering(self, env_ids: torch.Tensor):
        target_default_state = self.target_object.data.default_root_state[env_ids].clone()
        target_default_state[:, 7:] = 0.0
        target_default_state[:, 0:3] = (
            self.target_positions[env_ids] + self._terrain.env_origins[env_ids]
        )
        self.target_object.write_root_state_to_sim(target_default_state, env_ids)
