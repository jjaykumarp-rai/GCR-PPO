# throw_alpha_env.py
#
# Alpha throwing env (no G1 imports)
# Uses ThrowingAlphaGeneralEnvCfg from throw_alpha_env_cfg.py

from __future__ import annotations

import math
import copy
from math import gcd
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
from isaaclab.sim.utils import bind_visual_material

from .throw_alpha_env_cfg import ThrowingAlphaGeneralEnvCfg


TARGET_COLOR_PALETTE = (
    (0.894, 0.102, 0.110),
    (0.215, 0.494, 0.721),
    (0.302, 0.686, 0.290),
    (0.596, 0.306, 0.639),
    (1.000, 0.498, 0.000),
    (1.000, 1.000, 0.200),
    (0.651, 0.337, 0.157),
    (0.969, 0.506, 0.749),
    (0.600, 0.600, 0.600),
    (0.090, 0.745, 0.811),
)


class ThrowingAlphaEnv(DirectRLEnv):
    cfg: ThrowingAlphaGeneralEnvCfg

    def __init__(self, cfg: ThrowingAlphaGeneralEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Actions: (TJ1 + right arm + left arm) DOFs + 1 grip scalar
        # With TJ1 + 7 right + 7 left + 1 grip = 16 dims
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # sampled throwing commands and target positions
        self.throwing_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_positions = torch.zeros(self.num_envs, 3, device=self.device)

        # reward logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "throwing",
                "roll",
                "stability",
                "action_rate_l2",
                "dof_torques_l2",
                "dof_acc_l2",
            ]
        }

        # optional reward components
        self.reward_component_names = list(self._episode_sums.keys())
        self.reward_component_task_rew = ["throwing", "roll", "stability"]
        self.reward_components = len(self._episode_sums)

        if self.cfg.baseh_rew:
            self.reward_component_names.append("baseh_rew")
            self.reward_components += 1
            self.reward_component_task_rew.append("baseh_rew")
            self._episode_sums["baseh_rew"] = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.energy_rew:
            self.reward_component_names.append("energy_rew")
            self.reward_components += 1
            self.reward_component_task_rew.append("energy_rew")
            self._episode_sums["energy_rew"] = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.ballrel_rew:
            self.reward_component_names.append("ballrel_rew")
            self.reward_components += 1
            self.reward_component_task_rew.append("ballrel_rew")
            self._episode_sums["ballrel_rew"] = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.bodymo_rew:
            self.reward_component_names.append("bodymo_rew")
            self.reward_components += 1
            self.reward_component_task_rew.append("bodymo_rew")
            self._episode_sums["bodymo_rew"] = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.lftarm_rew:
            self.reward_component_names.append("lftarm_rew")
            self.reward_components += 1
            self.reward_component_task_rew.append("lftarm_rew")
            self._episode_sums["lftarm_rew"] = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.rgtarmrel_rew:
            self.reward_component_names.append("rgtarmrel_rew")
            self.reward_components += 1
            self.reward_component_task_rew.append("rgtarmrel_rew")
            self._episode_sums["rgtarmrel_rew"] = torch.zeros(self.num_envs, device=self.device)

        # ---- curriculum vectors (simple initialization) ----
        # These are used only if the corresponding *_rew flags are true.
        self.baseh_rew_vec = torch.zeros(1, device=self.device)
        self.energy_rew_vec = torch.zeros(1, device=self.device)
        self.ballrel_rew_vec = torch.zeros(1, device=self.device)
        self.bodymo_rew_vec = torch.zeros(1, device=self.device)
        self.lftarm_rew_vec = torch.zeros(1, device=self.device)
        self.rgtarmrel_rew_vec = torch.zeros(1, device=self.device)

        # ball release bookkeeping
        self.not_released_ball = torch.ones((self.num_envs), device=self.device).bool()
        self.throwing_reward = torch.ones((self.num_envs), device=self.device) * -1.0
        self.landing_time = torch.ones((self.num_envs), device=self.device) * -1.0
        self.throwing_reward_given = torch.zeros((self.num_envs), device=self.device).bool()
        self.stability_penalty_this_ep = torch.zeros((self.num_envs), device=self.device).bool()
        self.sum_open_hand_action = torch.zeros((self.num_envs), device=self.device)

        self.min_base_height = 0.38  # mostly irrelevant for Alpha, but kept for compatibility

        self.body_actions = 0.0 if self.cfg.arm_only else 1.0
        self.stability_timer = 2.0

        # ----- initial curriculum: start easy (close + low height), then go farther -----
        if self.cfg.distance_throw:
            # "pure distance" setting: start at 1 m and slowly push out
            self.distance_range = [1.0, 2.5]
            self.theta_range = [1.0, 1.0]
        else:
            # normal throwing: start close (1–3 m) and mostly flat
            self.distance_range = [1.0, 3.0]
            self.theta_range = [0.0, 0.3]

        self.target_half_fov_rad = min(math.radians(self.cfg.target_fov_deg) / 2.0, math.pi)
        self.target_heading_offset_rad = math.radians(self.cfg.target_heading_offset_deg)
        self.robot_yaw_offset_rad = math.radians(self.cfg.robot_yaw_offset_deg)

        self._sample_throwing_commands(torch.arange(self.num_envs, device=self.device))

        self.released_ball_t = torch.zeros((self.num_envs), device=self.device) - 1.0
        self.action_noise = 0.0

        self.ball_distances = torch.zeros((self.num_envs), device=self.device) - 1.0

        if self.cfg.no_proj_motion:
            self.cfg.obs_estimdisplace = False

        self.prev_velocity = torch.zeros((self.num_envs, 3), device=self.device) - 1000.0

        # default roots (for target placement helper)
        self.default_root_states = torch.zeros((self.num_envs, 3), device=self.device)

        # These will be filled in _setup_scene / after the first reset
        self._arm_dof_ids = None
        self._finger_dof_ids = None
        self._right_hand_body_id = None
        self._left_hand_body_id = None
        self._right_finger_body_ids = None
        self._hip_body_id = None
        self._hip_ids = None  # for compatibility with baseh/bodymo rewards

    # ----------------------------------------------------------------------
    # Scene setup
    # ----------------------------------------------------------------------

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

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self._apply_env_colors()

        # filter collisions with ground
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # lights
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        light_cfg.func("/World/Light", light_cfg, orientation=(1.0, 0.0, 0.0, 0.0))

        # Initialize DOF/body indices once simulation assets are loaded
        # (actually called lazily at first reset)
        # self._init_dof_and_body_indices()

    def _init_dof_and_body_indices(self):
        """Discover arm/finger DOFs and hand bodies by name substring (no regex failures)."""
        joint_names = list(self._robot.data.joint_names)
        body_names = list(self._robot.data.body_names)
        cs_body_names = list(self._contact_sensor.body_names)

        # Arm DOFs: trunk + right arm + left arm.
        # Adjust left_arm_joint_names if your naming is different.
        right_arm_joint_names = ["RJ1", "RJ2", "RJ3", "RJ4", "RJ5", "RJ6", "RJ7"]
        left_arm_joint_names = ["LJ1", "LJ2", "LJ3", "LJ4", "LJ5", "LJ6", "LJ7"]

        arm_joint_names = ["TJ1"] + right_arm_joint_names + left_arm_joint_names

        arm_ids = []
        for name in arm_joint_names:
            if name in joint_names:
                arm_ids.append(joint_names.index(name))
            else:
                print(f"[WARN] Arm joint '{name}' not found in joint_names.")

        self._arm_dof_ids = torch.tensor(arm_ids, device=self.device, dtype=torch.long)

        if len(arm_ids) != len(arm_joint_names):
            print(
                f"[WARN] Expected {len(arm_joint_names)} arm joints "
                f"(TJ1 + 7 right + 7 left), but found {len(arm_ids)}. "
                "Check joint naming in the Alpha URDF/USD."
            )

        # Finger DOFs: any joint name containing right_index / right_pinky / right_thumb
        finger_keywords = ["right_index", "right_pinky", "right_thumb"]
        finger_ids = [i for i, n in enumerate(joint_names) if any(k in n for k in finger_keywords)]
        self._finger_dof_ids = torch.tensor(finger_ids, device=self.device, dtype=torch.long)
        if len(finger_ids) == 0:
            print("[WARN] No finger DOFs detected for right hand; grip will be a no-op.")

        # Robot bodies: right/left palms and right finger links (for ball placement)
        def find_body_index(substring: str, default: int = 0):
            for i, n in enumerate(body_names):
                if substring in n:
                    return i
            print(f"[WARN] Could not find body containing '{substring}', defaulting to index {default}.")
            return default

        self._right_hand_body_id = find_body_index(
            "right_palm", default=body_names.index("RHAND") if "RHAND" in body_names else 0
        )
        self._left_hand_body_id = find_body_index(
            "left_palm", default=body_names.index("LHAND") if "LHAND" in body_names else 0
        )
        self._hip_body_id = find_body_index("PLINTH", default=0)
        self._hip_ids = torch.tensor([self._hip_body_id], device=self.device, dtype=torch.long)

        # right finger bodies for ball placement (use proximal segments)
        finger_body_keywords = ["right_index_proximal", "right_pinky_proximal", "right_thumb_proximal"]
        finger_body_ids = [find_body_index(k, default=self._right_hand_body_id) for k in finger_body_keywords]
        self._right_finger_body_ids = torch.tensor(finger_body_ids, device=self.device, dtype=torch.long)

        # Contact sensor groups: non-feet, hands, left/right palms
        def cs_indices_where(substr: str):
            return [i for i, n in enumerate(cs_body_names) if substr in n]

        self.non_feet_ids = torch.tensor(list(range(len(cs_body_names))), device=self.device, dtype=torch.long)
        self.hand_ids = torch.tensor(cs_indices_where("palm"), device=self.device, dtype=torch.long)
        left_ids = cs_indices_where("left_palm")
        right_ids = cs_indices_where("right_palm")
        self.left_hand_ids = torch.tensor(left_ids if len(left_ids) > 0 else [], device=self.device, dtype=torch.long)
        self.right_hand_ids = torch.tensor(right_ids if len(right_ids) > 0 else [], device=self.device, dtype=torch.long)

    # ----------------------------------------------------------------------
    # Color helpers
    # ----------------------------------------------------------------------

    def _apply_env_colors(self):
        """Apply per-env matching colors to ball and target (pairing like AlphaBallThrowEnv)."""
        palette_len = len(TARGET_COLOR_PALETTE)

        # Each env already has a root prim in scene.env_prim_paths like /World/envs/env_0
        for env_idx, env_path in enumerate(self.scene.env_prim_paths):
            color = TARGET_COLOR_PALETTE[env_idx % palette_len]

            # One material per env
            mat_path = f"/World/Visuals/env_{env_idx}_mat"
            mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
            mat_cfg.func(mat_path, mat_cfg)

            # Our prims from cfg:
            #   sphere_cfg.prim_path = "/World/envs/env_.*/sphere"
            #   target_cfg.prim_path = "/World/envs/env_.*/target"
            ball_path = f"{env_path}/sphere"
            target_path = f"{env_path}/target"

            bind_visual_material(ball_path, mat_path)
            bind_visual_material(target_path, mat_path)

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

    # ----------------------------------------------------------------------
    # RL step API
    # ----------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        # actions: [num_envs, num_arm_dofs + 1(grip)]
        self._actions = actions.clone()

        num_arm_dofs = self._arm_dof_ids.numel()

        # All arm DOFs (TJ1 + right + left arm)
        arm_actions = self._actions[:, :num_arm_dofs]  # [N, num_arm_dofs]
        default_arm_pos = self._robot.data.default_joint_pos[:, self._arm_dof_ids]  # [N, num_arm_dofs]
        arm_targets = default_arm_pos + self.cfg.action_scale * arm_actions

        # Next dim is grip: >0 = open, <0 = closed
        grip = self._actions[:, num_arm_dofs].unsqueeze(-1)  # [N, 1]

        # finger DOFs
        if self._finger_dof_ids is not None and self._finger_dof_ids.numel() > 0:
            default_finger_pos = self._robot.data.default_joint_pos[:, self._finger_dof_ids]  # [N, F]
            closed_finger_pos = default_finger_pos + 0.5  # tune
            open_finger_pos = default_finger_pos - 0.2   # tune

            alpha = (grip.clamp(-1.0, 1.0) + 1.0) * 0.5  # [-1,1] → [0,1]
            finger_targets = (1 - alpha) * closed_finger_pos + alpha * open_finger_pos
        else:
            finger_targets = None

        self._arm_targets = arm_targets
        self._finger_targets = finger_targets

        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(torch.arange(self.num_envs, device=self.device))

    def _apply_action(self):
        # Set arm targets
        self._robot.set_joint_position_target(self._arm_targets, joint_ids=self._arm_dof_ids)
        # Set finger targets (if any)
        if self._finger_targets is not None and self._finger_dof_ids.numel() > 0:
            self._robot.set_joint_position_target(self._finger_targets, joint_ids=self._finger_dof_ids)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel

        joint_pos_info = joint_pos
        joint_vel_info = joint_vel

        # ball displacement estimate
        estimated_displacement, estim_time = self.check_ball_displacement(
            torch.arange(self.num_envs, device=self.device)
        )
        estimated_displacement = 1.0 - estimated_displacement

        # base roll
        w, x, y, z = self._robot.data.root_quat_w.T
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
        roll = torch.where(torch.isnan(roll), torch.zeros_like(roll), roll)

        noise_displace = (1.0 - self.action_noise) * estimated_displacement + self.action_noise * torch.randn_like(
            estimated_displacement
        )

        self.prev_velocity = self._robot.data.root_lin_vel_b.clone()
        ball_data = self.sphere_object.data.body_state_w[:, 0, [7, 8, 9]]  # linear vel

        obs_tensors = []

        if self.cfg.obs_ang_vel:
            obs_tensors.append(self._robot.data.root_ang_vel_b)
        if self.cfg.obs_proj_grav:
            obs_tensors.append(self._robot.data.projected_gravity_b)
        if self.cfg.obs_roll:
            obs_tensors.append((roll.float() + (torch.rand_like(roll) * 0.02 - 0.01)).unsqueeze(-1))

        obs_tensors.append(self.throwing_commands)
        obs_tensors.append(joint_pos_info + (torch.rand_like(joint_pos_info) * 0.02 - 0.01))
        obs_tensors.append(joint_vel_info + (torch.rand_like(joint_vel_info) * 0.1 - 0.05))
        obs_tensors.append(self._actions)

        if self.cfg.obs_notrelease:
            obs_tensors.append(self.not_released_ball.unsqueeze(-1).float())
        if self.cfg.obs_estimdisplace:
            obs_tensors.append(noise_displace.unsqueeze(-1))
            obs_tensors.append(estim_time.unsqueeze(-1))
            obs_tensors.append(ball_data)

        obs = torch.cat(obs_tensors, dim=-1)
        obs = torch.clip(obs, -1000.0, 1000.0)

        if torch.sum(torch.isnan(obs)) > 0:
            print("[WARN] NaNs in obs:", torch.sum(torch.isnan(obs), dim=0))

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # --- common: distance from ball to right hand (release detector) ---
        ball_pos = self.sphere_object.data.body_pos_w[:, 0, 0:3]
        right_hand_pos = self._robot.data.body_pos_w[:, self._right_hand_body_id, :3]
        hand_ball_dist = torch.norm(ball_pos - right_hand_pos, dim=1)

        if self.cfg.no_proj_motion:
            # "no projectile" variant (kept for compatibility)
            throwing_reward_condition = (hand_ball_dist > 0.25)

            target_positions = self.target_positions[:] + self._terrain.env_origins[:]
            targ_dist = torch.norm(ball_pos - target_positions, dim=1).reshape(self.num_envs)
            on_ground = self.sphere_object.data.body_pos_w[:, 0, 2] < 0.05

            self.ball_distances[:] = torch.where(
                ~on_ground
                & ((self.ball_distances[:] == -1) | (targ_dist < self.ball_distances[:])),
                targ_dist,
                self.ball_distances[:],
            )
            env_ids = torch.nonzero((self.reset_buf == 1) & throwing_reward_condition).reshape(-1)
            self.throwing_reward[:] = 0.0
            if len(env_ids) > 0:
                self.throwing_reward[env_ids] = self.ball_distances[env_ids] / self.throwing_commands[env_ids, 0]
                self.throwing_reward[env_ids] = 1.0 - torch.min(
                    torch.tensor(1.0, device=self.device), self.throwing_reward[env_ids]
                )
        else:
            # normal ballistic mode: detect "release" when ball leaves hand
            throwing_reward_condition = (hand_ball_dist > 0.25) & (~self.throwing_reward_given)
            self.not_released_ball &= (hand_ball_dist <= 0.25)

            env_ids = torch.nonzero(throwing_reward_condition).reshape(-1)
            self.throwing_reward[:] = 0.0
            if len(env_ids) > 0:
                self.throwing_reward[env_ids], self.landing_time[env_ids] = self.check_ball_displacement(env_ids)
                # distance is normalized by commanded distance → 0 = perfect, 1 = far off
                self.throwing_reward[env_ids] = 1.0 - self.throwing_reward[env_ids]
                self.throwing_reward_given[env_ids] = True
                self.landing_time[env_ids] += self.episode_length_buf[env_ids] * self.step_dt

        ball_released_envs = env_ids

        # roll reward
        w, x, y, z = self._robot.data.root_quat_w.T
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        roll = torch.where(torch.isnan(roll), torch.zeros_like(roll), roll)
        roll_rew = (-1.0 / (1.0 + torch.exp(-10.0 * (torch.abs(roll) - 0.3)))) * (
            1.0 - torch.exp(-(torch.abs(roll) - 0.1) / 0.1)
        )
        if not self.cfg.use_roll_reward:
            roll_rew = torch.zeros_like(roll_rew)

        # stability reward
        base_height_cond = self._robot.data.root_pos_w[:, 2] <= self.min_base_height
        hand_positions = self._robot.data.body_pos_w[:, [-1], :].reshape(self.num_envs, 3)
        ball_positions = self.sphere_object.data.root_pos_w.clone()
        ball_not_thrown_cond = (torch.norm((hand_positions - ball_positions), dim=1) <= 0.25) & (
            self.reset_buf == 1
        )

        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)
        mask = torch.where(
            torch.norm((hand_positions - ball_positions), dim=1) <= 0.25,
            torch.zeros(1, device=self.device),
            torch.ones(1, device=self.device),
        )

        first_contact[:, self.hand_ids] &= mask.view(-1, 1).expand(-1, len(self.hand_ids)).to(torch.bool)
        collision = torch.sum(first_contact[:, self.non_feet_ids], dim=1) > 0

        self.stability_penalty_this_ep |= (ball_not_thrown_cond | collision)
        stability_rew = ((self.reset_buf == 1) & (~self.stability_penalty_this_ep)).float()
        if self.cfg.nonsparse_stability_reward:
            stability_rew = (
                (~(base_height_cond | ball_not_thrown_cond | collision)).float()
                / self.max_episode_length_s
            ) * self.step_dt
        if not self.cfg.use_stability_reward:
            stability_rew = torch.zeros_like(stability_rew)

        # regularization terms
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

        rewards = {
            "throwing": torch.clamp(self.throwing_reward.clone(), min=0.0)
            * self.cfg.throwing_reward_scale
            * self.max_episode_length_s,
            "roll": roll_rew * self.cfg.roll_reward_scale * self.step_dt,
            "stability": stability_rew * self.cfg.stability_reward_scale * self.max_episode_length_s,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
        }

        # ---- optional curriculum rewards (unchanged logic, just added into rewards dict) ----
        r_scale = 1.0

        if self.cfg.baseh_rew:
            base_height = self._robot.data.body_pos_w[:, self._hip_ids[0], 2]
            bounds_list = [[0.35, 0.4], [0.4, 0.45], [0.45, 0.5], [0.5, 0.55], [0.55, 0.6]]
            selected_bounds = bounds_list[self.baseh_rew_vec[0].int()]
            min_value, max_value = selected_bounds[0], selected_bounds[1]
            reward_term = ((base_height >= min_value) & (base_height <= max_value)).float()
            reward_term *= self.step_dt
            rewards["baseh_rew"] = reward_term
            within = (base_height >= min_value) & (base_height <= max_value)
            den = torch.ones_like(within).sum()
            pct = 100.0 * within.float().sum() / den
            self.extras["log"]["base_height_success_pct"] = pct

        if self.cfg.energy_rew:
            # FIXED: use only actuated torso+arms joints (self._arm_dof_ids)
            num_arm_dofs = self._arm_dof_ids.numel()
            joint_vel_arm = self._robot.data.joint_vel[:, self._arm_dof_ids]  # [N, num_arm_dofs]
            electricity_cost = torch.sum(
                torch.abs(self._actions[:, :num_arm_dofs] * joint_vel_arm),
                dim=-1,
            )
            energy_vals_tensor = [[40, 70], [70, 100], [100, 130], [130, 160], [160, 200]]
            selected_bounds = energy_vals_tensor[self.energy_rew_vec[0].int()]
            min_energy = selected_bounds[0]
            max_energy = selected_bounds[1]
            reward_electricity = ((electricity_cost >= min_energy) & (electricity_cost <= max_energy)).float()
            reward_electricity *= self.step_dt * r_scale
            rewards["energy_rew"] = reward_electricity
            within = (electricity_cost >= min_energy) & (electricity_cost <= max_energy)
            den = torch.ones_like(within).sum()
            pct = 100.0 * within.float().sum() / den
            self.extras["log"]["energy_success_pct"] = pct

        if self.cfg.ballrel_rew:
            if len(ball_released_envs) == 0:
                reward_term_full = torch.zeros(self.num_envs, device=self.device)
                rewards["ballrel_rew"] = reward_term_full
                self.extras["log"]["ballrel_success_pct"] = torch.tensor(0.0, device=self.device)
            else:
                ball_released_time = self.episode_length_buf[ball_released_envs].float() * self.step_dt
                measured_val = ball_released_time
                bounds_list = [[0.0, 0.3], [0.3, 0.6], [0.6, 0.9], [0.9, 1.2], [1.2, 1.5]]
                selected_bounds = bounds_list[self.ballrel_rew_vec[0].int()]
                min_value, max_value = selected_bounds[0], selected_bounds[1]
                reward_term = ((measured_val >= min_value) & (measured_val <= max_value)).float()
                reward_term *= r_scale
                reward_term_full = torch.zeros(self.num_envs, device=self.device)
                reward_term_full[ball_released_envs] = reward_term
                rewards["ballrel_rew"] = reward_term_full
                within = (measured_val >= min_value) & (measured_val <= max_value)
                den = torch.ones_like(within).sum()
                pct = 100.0 * within.float().sum() / den
                self.extras["log"]["ballrel_success_pct"] = pct

        if self.cfg.bodymo_rew:
            if len(ball_released_envs) == 0:
                reward_term = torch.zeros(self.num_envs, device=self.device)
                rewards["bodymo_rew"] = reward_term
                self.extras["log"]["bodymo_success_pct"] = torch.tensor(0.0, device=self.device)
            else:
                measured_val = torch.norm(
                    self._robot.data.body_vel_w[:, self._hip_ids[0], :3],
                    dim=-1,
                )
                bounds_list = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
                selected_bounds = bounds_list[self.bodymo_rew_vec[0].int()]
                min_value, max_value = selected_bounds[0], selected_bounds[1]
                reward_term = ((measured_val >= min_value) & (measured_val <= max_value)).float()
                reward_term *= r_scale
                reward_term_full = torch.zeros(self.num_envs, device=self.device)
                reward_term_full[ball_released_envs] = reward_term[ball_released_envs]
                rewards["bodymo_rew"] = reward_term_full
                within = (measured_val >= min_value) & (measured_val <= max_value)
                den = torch.ones_like(within).sum()
                pct = 100.0 * within.float().sum() / den
                self.extras["log"]["bodymo_success_pct"] = pct

        if self.cfg.lftarm_rew:
            measured_val = self._robot.data.body_pos_w[:, self.left_hand_ids[0], 2]
            bounds_list = [[0.6, 0.8], [0.8, 1.0], [1.0, 1.2], [1.2, 1.4], [1.4, 1.6]]
            selected_bounds = bounds_list[self.lftarm_rew_vec[0].int()]
            min_value, max_value = selected_bounds[0], selected_bounds[1]
            reward_term = ((measured_val >= min_value) & (measured_val <= max_value)).float()
            reward_term *= self.step_dt * r_scale
            rewards["lftarm_rew"] = reward_term
            within = (measured_val >= min_value) & (measured_val <= max_value)
            den = torch.ones_like(within).sum()
            pct = 100.0 * within.float().sum() / den
            self.extras["log"]["lftarm_success_pct"] = pct

        if self.cfg.rgtarmrel_rew:
            if len(ball_released_envs) == 0:
                reward_term = torch.zeros(self.num_envs, device=self.device)
                rewards["rgtarmrel_rew"] = reward_term
                self.extras["log"]["rgtarmrel_success_pct"] = torch.tensor(0.0, device=self.device)
            else:
                measured_val = self._robot.data.body_pos_w[:, self.right_hand_ids[0], 2]
                bounds_list = [[0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8]]
                selected_bounds = bounds_list[self.rgtarmrel_rew_vec[0].int()]
                min_value, max_value = selected_bounds[0], selected_bounds[1]
                reward_term = ((measured_val >= min_value) & (measured_val <= max_value)).float()
                reward_term *= r_scale
                reward_term_full = torch.zeros(self.num_envs, device=self.device)
                reward_term_full[ball_released_envs] = reward_term[ball_released_envs]
                rewards["rgtarmrel_rew"] = reward_term_full
                within = (measured_val >= min_value) & (measured_val <= max_value)
                den = torch.ones_like(within).sum()
                pct = 100.0 * within.float().sum() / den
                self.extras["log"]["rgtarmrel_success_pct"] = pct

        # ---- combine into scalar reward per env (RSL-RL expects 1D rewards) ----
        reward_components = torch.stack(list(rewards.values()), dim=1)  # [N, num_components]
        total_reward = torch.sum(reward_components, dim=1)  # [N]

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return total_reward

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Initialize DOF/body indices once, now that articulation.data exists
        if self._arm_dof_ids is None:
            self._init_dof_and_body_indices()

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self.body_actions = 0.0 if self.cfg.arm_only else 1.0
        if self.cfg.distance_throw:
            self.distance_range = [max(self.distance_range[0], 4.0), max(self.distance_range[1], 4.0)]

        if self.cfg.no_proj_motion:
            self.cfg.obs_estimdisplace = False

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

        # reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
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
        self.default_root_states[env_ids] = default_root_state[:, :3]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(env_ids)

        # initialise ball in hand (average right finger body positions)
        object_default_state = self.sphere_object.data.default_root_state.clone()[env_ids]
        finger_positions = self._robot.data.body_pos_w[:, self._right_finger_body_ids, :][env_ids]
        finger_positions = torch.mean(finger_positions, dim=1)  # [len(env_ids), 3]
        hand_positions = self._robot.data.body_pos_w[:, self._right_hand_body_id, :][env_ids]  # [len,3]

        vector_AB = finger_positions - hand_positions
        distance_AB = torch.norm(vector_AB, dim=1).reshape(len(env_ids), 1)
        direction = vector_AB / torch.clamp(distance_AB, min=1e-6)
        distance_AC = 0.0 * distance_AB
        position_c = finger_positions  # hand_positions + direction * distance_AC

        object_default_state[:, 0:3] += position_c
        object_default_state[:, 7:] = self._robot.data.body_vel_w[:, self._right_hand_body_id, :][env_ids].reshape(
            len(env_ids), 6
        )
        self.sphere_object.write_root_state_to_sim(object_default_state, env_ids)

        if self.cfg.no_proj_motion:
            self.ball_distances[env_ids] = -1.0
        self.prev_velocity[env_ids] = torch.zeros((len(env_ids), 3), device=self.device) - 1000.0

        # Logging
        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
            if "stability" in key:
                extras["Episode_Reward/" + key] /= self.cfg.stability_reward_scale
            if "throwing" in key:
                extras["Episode_Reward/" + key] /= self.cfg.throwing_reward_scale
            if "roll" in key:
                extras["Episode_Reward/" + key] /= self.cfg.roll_reward_scale

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras2 = dict()
        extras2["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras2["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras2["Target_Distance"] = max(self.distance_range)
        extras2["Theta_Range_Max"] = max(self.theta_range)
        extras2["Body_Actions"] = self.body_actions
        extras2["Stability_Timer"] = self.stability_timer
        self.extras["log"].update(extras2)

    # ----------------------------------------------------------------------
    # Curriculum + helpers
    # ----------------------------------------------------------------------
    def update_curriculum(self, iter: int):
        """Easy → hard: start close, then gradually increase distance and elevation spread."""
        stability_value = self.extras["log"]["Episode_Reward/stability"]
        throwing_value = self.extras["log"]["Episode_Reward/throwing"]

        # gate: only advance if we're doing well enough, or after many iterations
        success = (throwing_value > self.cfg.r_throw_thresh) and (stability_value > self.cfg.r_stability_thresh)
        if not success and iter < 500:
            return

        # how much to expand per curriculum step
        dist_step_max = 0.05   # push target farther by 5 cm (max distance)
        dist_step_min = 0.02   # also slowly move min distance to keep window width reasonable
        theta_step = 0.02      # widen elevation band

        # current ranges
        d_min, d_max = self.distance_range

        # grow max distance up to cfg.max_throw_dist
        new_d_max = min(self.cfg.max_throw_dist, d_max + dist_step_max)
        # keep at least 1 m window, move min up slowly but never beyond new_d_max - 0.5
        new_d_min = min(new_d_max - 0.5, d_min + dist_step_min)
        new_d_min = max(0.5, new_d_min)  # never let it go too close to zero

        self.distance_range = [float(new_d_min), float(new_d_max)]

        # widen theta range towards full [0, 1]
        self.theta_range = [
            self.theta_range[0],
            float(min(1.0, self.theta_range[1] + theta_step)),
        ]

    def _commands_to_target(self, commands: torch.Tensor) -> torch.Tensor:
        """Convert sampled polar commands into cartesian offsets (forward = +Y)."""
        dist = commands[:, 0]
        theta_cos = commands[:, 1]
        yaw = commands[:, 2]

        # Recover height from distance and theta cosine.
        height = theta_cos * dist

        # Alpha faces +Y, so yaw=0 points along +Y.
        x = dist * torch.sin(yaw)   # sideways
        y = dist * torch.cos(yaw)   # forward
        z = height

        return torch.stack((x, y, z), dim=-1)

    def _sample_throwing_commands(self, env_ids: torch.Tensor | None):
        """Sample target distance/yaw/height in a forward FOV in front of Alpha (+Y)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        command_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Distances / heights (same semantics as before)
        dist_min, dist_max = self.distance_range
        z_min, z_max = self.cfg.target_height_range
        margin = 0.05

        # FOV around "forward" (Alpha faces +Y, yaw=0 => +Y)
        fov_rad = math.radians(max(0.0, min(360.0, self.cfg.target_fov_deg)))
        half_span = fov_rad / 2.0 if fov_rad > 0.0 else math.pi

        # yaw within [-half_span, +half_span]
        yaw = torch.zeros(len(command_ids), device=self.device).uniform_(-half_span, half_span)

        # sample height and distance
        z_samples = torch.zeros(len(command_ids), device=self.device).uniform_(z_min, z_max)
        z_clamped = torch.clamp(z_samples, min=0.0, max=dist_max - margin)

        dist_samples = torch.zeros(len(command_ids), device=self.device).uniform_(dist_min, dist_max)
        dist_final = torch.max(dist_samples, z_clamped + margin)

        theta_cos = torch.clamp(z_clamped / dist_final, -1.0, 1.0)

        if self.cfg.distance_throw:
            # For distance-throw experiments: flat, straight ahead
            theta_cos = torch.ones_like(theta_cos)
            yaw = torch.zeros_like(yaw)

        commands = torch.stack((dist_final, theta_cos, yaw), dim=-1)
        self.throwing_commands[command_ids] = commands

        offsets = self._commands_to_target(commands)
        self.target_positions[env_ids] = offsets

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

    def calculate_target_offset(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Wrapper to keep old API; uses +Y-forward commands→target mapping."""
        command_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        commands = self.throwing_commands[command_ids]
        return self._commands_to_target(commands)

    def check_ball_displacement(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ball_data = self.sphere_object.data.body_state_w[env_ids, 0, :10]

        # Use the SAME world target as rendering: env_origin + offset
        target_positions = self.target_positions[env_ids] + self._terrain.env_origins[env_ids]
        # IMPORTANT: do NOT add default_root_states again (that was double-counting)

        if self.cfg.air_resistance:
            raise NotImplementedError

        distance_command = self.throwing_commands[env_ids, 0].unsqueeze(0)
        vx0 = ball_data[:, 7]
        vy0 = ball_data[:, 8]
        vz0 = ball_data[:, 9]
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

    def initialise_target_for_rendering(self, env_ids: torch.Tensor):
        # Clone default state for these envs
        target_default_state = self.target_object.data.default_root_state.clone()[env_ids]

        # Zero linear + angular velocities
        target_default_state[:, 7:] = 0.0

        # World-space position of the target: env origin + sampled offset
        world_target_pos = self.target_positions[env_ids] + self._terrain.env_origins[env_ids]
        target_default_state[:, 0:3] = world_target_pos

        # ================================
        # Orient cylinder plane to face robot
        # ================================
        # Robot base position (world)
        robot_pos = self._robot.data.root_pos_w[env_ids, :3]

        # Desired normal: direction from target -> robot
        direction_to_robot = robot_pos - world_target_pos
        direction_to_robot = F.normalize(direction_to_robot, p=2, dim=1)

        # CylinderCfg(axis="X") => local +X is face normal
        # So we rotate world +X to direction_to_robot
        forward_vector = torch.tensor(
            [1.0, 0.0, 0.0], device=target_default_state.device
        ).view(1, 3).expand_as(direction_to_robot)

        # Axis of rotation: cross(forward, desired)
        axis = torch.cross(forward_vector, direction_to_robot, dim=1)
        axis_norm = torch.norm(axis, dim=1, keepdim=True)

        # Handle near-parallel case: if axis ~ 0, pick a fallback axis (e.g., Z)
        fallback_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(1, 3).expand_as(axis)
        axis = torch.where(axis_norm < 1e-6, fallback_axis, axis / axis_norm.clamp(min=1e-6))

        # Angle between forward and direction_to_robot
        dot = (forward_vector * direction_to_robot).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        angle = torch.acos(dot)

        # Build quaternion from axis-angle
        sin_half = torch.sin(angle / 2.0)
        xyz = axis * sin_half
        w = torch.cos(angle / 2.0)
        quat_delta = torch.cat([w, xyz], dim=1)  # [w, x, y, z]

        # Multiply with default orientation
        current_quaternion = target_default_state[:, 3:7]

        def quaternion_multiply(q1, q2):
            w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
            w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return torch.stack([w, x, y, z], dim=1)

        new_quaternion = quaternion_multiply(current_quaternion, quat_delta)
        target_default_state[:, 3:7] = new_quaternion

        # Write back to sim
        self.target_object.write_root_state_to_sim(target_default_state, env_ids)
