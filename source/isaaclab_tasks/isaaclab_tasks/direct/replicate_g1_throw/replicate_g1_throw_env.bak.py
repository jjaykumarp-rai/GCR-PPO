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
from .alpha_utils import (
    TORSO_JOINTS,
    RIGHT_ARM_JOINTS,
    LEFT_ARM_JOINTS,
    FINGER_JOINTS,
)


class ReplicateG1ThrowEnv(DirectRLEnv):
    cfg: ReplicateG1ThrowEnvCfg

    def __init__(self, cfg: ReplicateG1ThrowEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        self.throwing_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_positions = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "throwing",
                # "roll",
                # "stability",
                "action_rate_l2",
                "dof_torques_l2",
                "dof_acc_l2",
            ]
        }

        # Track best distance (used only if no_proj_motion=True)
        self.ball_distances = torch.ones(self.num_envs, device=self.device) * -1.0

        # Stability flag (even if we don't use stability reward, avoid attribute errors)
        self.stability_penalty_this_ep = torch.zeros(self.num_envs, device=self.device).bool()

        self.reward_components = len(self._episode_sums.keys())
        self.reward_component_names = list(self._episode_sums.keys())
        self.reward_component_task_rew = ["throwing"]

        if getattr(self.cfg, "baseh_rew", False):
            self.reward_component_names += ["baseh_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["baseh_rew"]
            self._episode_sums["baseh_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if getattr(self.cfg, "energy_rew", False):
            self.reward_component_names += ["energy_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["energy_rew"]
            self._episode_sums["energy_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if getattr(self.cfg, "ballrel_rew", False):
            self.reward_component_names += ["ballrel_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["ballrel_rew"]
            self._episode_sums["ballrel_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if getattr(self.cfg, "bodymo_rew", False):
            self.reward_component_names += ["bodymo_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["bodymo_rew"]
            self._episode_sums["bodymo_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if getattr(self.cfg, "lftarm_rew", False):
            self.reward_component_names += ["lftarm_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["lftarm_rew"]
            self._episode_sums["lftarm_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if getattr(self.cfg, "rgtarmrel_rew", False):
            self.reward_component_names += ["rgtarmrel_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["rgtarmrel_rew"]
            self._episode_sums["rgtarmrel_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.not_released_ball = torch.ones((self.num_envs), device=self.device).bool()
        self.throwing_reward = torch.ones((self.num_envs), device=self.device)*-1 # by default it is -1 if not thrown
        self.landing_time = torch.ones((self.num_envs), device=self.device)*-1
        self.throwing_reward_given = torch.zeros((self.num_envs), device=self.device).bool()
        self.sum_open_hand_action = torch.zeros((self.num_envs), device=self.device)
        
        # removed stability related things
        # self.min_base_height = 0.38 
        # self.stability_penalty_this_ep = torch.zeros((self.num_envs), device=self.device).bool()

        if self.cfg.arm_only:
            self.body_actions = 0.
        else:
            self.body_actions = 1. # max is 100%

        # removed stability related things
        # self.stability_timer = 2. # max is 2 (seconds)

        if self.cfg.distance_throw:
            self.distance_range = [4., 4.]
            self.theta_range = [1.0, 1.0]
        else:
            self.distance_range = [4., 8.] 
            self.theta_range = [0., 1.]

        self.target_half_fov_rad = min(math.radians(self.cfg.target_fov_deg) / 2.0, math.pi)
        self.target_heading_offset_rad = math.radians(self.cfg.target_heading_offset_deg)
        self.robot_yaw_offset_rad = math.radians(self.cfg.robot_yaw_offset_deg)
        self._sample_throwing_commands(torch.arange(self.num_envs, device=self.device))
        self.released_ball_t = torch.zeros((self.num_envs), device=self.device)-1
        self.action_noise = 0.

        if self.cfg.no_proj_motion:
            self.cfg.obs_estimdisplace = False

        self.prev_velocity = torch.zeros((self.num_envs, 3), device=self.device)-1000
        
        self.default_root_states = torch.zeros((self.num_envs, 3), device=self.device)

    def _sample_throwing_commands(self, env_ids: torch.Tensor) -> None:
        """Sample per-env throwing commands: [distance, heading, theta_param].

        distance: radial distance of target [m]
        heading:  yaw offset within a FOV (radians)
        theta_param: generic [0,1] parameter (kept from G1 version, can be used later)
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        num = env_ids.shape[0]

        # distance in [distance_range[0], distance_range[1]]
        distances = sample_uniform(
            self.distance_range[0],
            self.distance_range[1],
            (num, 1),
            device=self.device,
        )

        # heading within +/- target_half_fov_rad (around 0)
        headings = sample_uniform(
            -self.target_half_fov_rad,
            self.target_half_fov_rad,
            (num, 1),
            device=self.device,
        )

        # generic theta parameter in [theta_range[0], theta_range[1]]
        thetas = sample_uniform(
            self.theta_range[0],
            self.theta_range[1],
            (num, 1),
            device=self.device,
        )

        cmds = torch.cat([distances, headings, thetas], dim=-1)
        self.throwing_commands[env_ids] = cmds


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


    def check_ball_displacement(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate normalized distance error and a dummy landing time.

        Returns:
            displacement_norm: in [0, 1], 0 is perfect hit, 1 is far off.
            landing_time: dummy zeros for now (can be replaced with true TOF later).
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return (
                torch.zeros(0, device=self.device),
                torch.zeros(0, device=self.device),
            )

        # current ball position (world)
        ball_pos = self.sphere_object.data.body_pos_w[env_ids, 0, 0:3]

        # target position (world): env origin + per-env target offset
        target_pos = self._terrain.env_origins[env_ids] + self.target_positions[env_ids]

        # Euclidean distance
        dist = torch.norm(ball_pos - target_pos, dim=-1)

        # Normalize by commanded distance (avoid division by zero)
        commanded_dist = self.throwing_commands[env_ids, 0].abs().clamp(min=1e-3)
        displacement_norm = torch.clamp(dist / commanded_dist, max=1.0)

        # Placeholder: no true time-of-flight yet
        landing_time = torch.zeros_like(displacement_norm)

        return displacement_norm, landing_time

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiply q = q1 * q2, both (..., 4) in (w, x, y, z) format."""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack((w, x, y, z), dim=-1)

    def initialise_target_for_rendering(self, env_ids: torch.Tensor) -> None:
        """Position the visual target in GUI based on throwing_commands."""
        if not hasattr(self, "target_object"):
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        # target world position = env origin + target offset
        target_pos = self._terrain.env_origins[env_ids] + self.target_positions[env_ids]  # (N, 3)

        # identity orientation
        target_quat = torch.zeros((env_ids.numel(), 4), device=self.device)
        target_quat[:, 0] = 1.0  # w = 1

        # pack into (N, 7) root state
        root_state = torch.zeros((env_ids.numel(), 7), device=self.device)
        root_state[:, :3] = target_pos
        root_state[:, 3:] = target_quat

        # RigidObject API: (root_state, env_ids)
        self.target_object.write_root_pose_to_sim(root_state, env_ids)



    def _init_dof_groups(self):
        if hasattr(self, "hand_dof_indices"):
            return

        joint_names = list(self._robot.data.joint_names)

        # indices for non-finger joints (torso + arms)
        non_finger_names = set(TORSO_JOINTS + RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS)
        finger_names = set(FINGER_JOINTS)

        self.non_finger_dof_indices = torch.tensor(
            [i for i, n in enumerate(joint_names) if n in non_finger_names],
            device=self.device,
            dtype=torch.long,
        )
        self.finger_dof_indices = torch.tensor(
            [i for i, n in enumerate(joint_names) if n in finger_names],
            device=self.device,
            dtype=torch.long,
        )
    
    def _cache_dof_indices(self):
        """Cache DOF indices for torso+arms and finger joints based on joint names."""
        if hasattr(self, "_arm_dof_indices"):
            return  # already done

        joint_names = list(self._robot.data.joint_names)
        name_to_index = {name: i for i, name in enumerate(joint_names)}

        # Arm (torso + both arms) joints in the same order as your action vector
        self._arm_joint_names = TORSO_JOINTS + RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS
        self._arm_dof_indices = torch.tensor(
            [name_to_index[n] for n in self._arm_joint_names],
            device=self.device,
            dtype=torch.long,
        )

        # Finger joints (right + left)
        self._finger_dof_indices = torch.tensor(
            [name_to_index[n] for n in FINGER_JOINTS],
            device=self.device,
            dtype=torch.long,
        )


    def _cache_alpha_indices(self):
        """Cache DOF indices (torso/arms/fingers) and body/contact indices for Alpha."""
        if hasattr(self, "_alpha_indices_cached") and self._alpha_indices_cached:
            return

        # -------------------------
        # Joint DOF indices
        # -------------------------
        joint_names = list(self._robot.data.joint_names)
        jname_to_idx = {n: i for i, n in enumerate(joint_names)}

        # Arm (torso + both arms) joints in the same order as your action vector
        self._arm_joint_names = TORSO_JOINTS + RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS
        self._arm_dof_indices = torch.tensor(
            [jname_to_idx[n] for n in self._arm_joint_names],
            device=self.device,
            dtype=torch.long,
        )

        # Finger joints
        self._finger_dof_indices = torch.tensor(
            [jname_to_idx[n] for n in FINGER_JOINTS],
            device=self.device,
            dtype=torch.long,
        )

        # -------------------------
        # Body indices (for rewards)
        # -------------------------
        body_names = list(self._robot.data.body_names)
        bname_to_idx = {n: i for i, n in enumerate(body_names)}

        # Root / pelvis / hip body – fall back to body 0 if we can't find a better name
        self.hip_body_id = (
            bname_to_idx.get("HIP")
            or bname_to_idx.get("hip")
            or bname_to_idx.get("pelvis")
            or 0
        )

        # Hands – try multiple reasonable names
        self.right_hand_body_id = (
            bname_to_idx.get("RHAND")
            or bname_to_idx.get("right_hand")
            or bname_to_idx.get("right_hand_link")
            or 0
        )
        self.left_hand_body_id = (
            bname_to_idx.get("LHAND")
            or bname_to_idx.get("left_hand")
            or bname_to_idx.get("left_hand_link")
            or 0
        )

        # -------------------------
        # Contact sensor body groups
        # -------------------------
        # "non-feet" = all sensor bodies (Alpha is fixed base, so no ankles anyway)
        all_ids, _ = self._contact_sensor.find_bodies(".*")
        self.non_feet_ids = all_ids

        # All hand/finger bodies
        self.hand_ids, _ = self._contact_sensor.find_bodies(".*(HAND|hand|finger|thumb).*")

        # Left / right hand subsets (if present)
        self.left_hand_ids, _ = self._contact_sensor.find_bodies(".*(LHAND|left_hand).*")
        self.right_hand_ids, _ = self._contact_sensor.find_bodies(".*(RHAND|right_hand).*")

        self._alpha_indices_cached = True


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
        # Cache indices once
        self._cache_alpha_indices()

        # store incoming policy actions
        self._actions = actions.clone()
        self.actions = actions.clone()  # if you still want this

        # First part: torso + right arm + left arm (in _arm_joint_names order)
        num_arm_dofs = self._arm_dof_indices.shape[0]
        arm_actions = self._actions[:, :num_arm_dofs]          # (N_envs, N_arm_dofs)
        hand_scalar = self._actions[:, -1].view(-1, 1)         # (N_envs, 1)

        # Start from default joint positions
        default_positions = self._robot.data.default_joint_pos.clone()
        target_positions = default_positions.clone()

        # Relative offsets for torso + arms
        target_positions[:, self._arm_dof_indices] = (
            default_positions[:, self._arm_dof_indices]
            + self.cfg.action_scale * arm_actions
        )

        # -------------------------
        # Finger actions (Alpha hand)
        # -------------------------
        # Keep existing semantics:
        #   hand_scalar >= 0 -> hand closed
        #   hand_scalar < 0  -> hand open
        open_angle = 0.0    # radians
        closed_angle = 0.7  # radians (tune this!)

        closed_mask = (hand_scalar >= 0.0).float()  # (N_envs, 1)
        finger_target = closed_mask * closed_angle + (1.0 - closed_mask) * open_angle
        target_positions[:, self._finger_dof_indices] = finger_target

        # Save processed joint targets
        self._processed_actions = target_positions

        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(torch.arange(self.num_envs, device=self.device))

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._processed_actions)#[:,:-1])

    def _get_observations(self) -> dict:
        # Cache indices
        self._cache_alpha_indices()

        self._previous_actions = self._actions.clone()

        # Joint pos/vel for torso + both arms
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        default_joint_pos = self._robot.data.default_joint_pos

        joint_pos_info = (joint_pos - default_joint_pos)[:, self._arm_dof_indices]
        joint_vel_info = joint_vel[:, self._arm_dof_indices]

        # Ball displacement + time estimate
        env_ids = torch.arange(self.num_envs, device=self.device)
        estimated_displacement, estim_time = self.check_ball_displacement(env_ids)
        estimated_displacement = 1.0 - estimated_displacement

        # Root roll (for obs_roll)
        w, x, y, z = self._robot.data.root_quat_w.T
        roll = torch.atan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )

        # Add small noise (as in original)
        roll_noisy = roll.float() + (torch.rand_like(roll) * 0.02 - 0.01)
        joint_pos_noisy = joint_pos_info + (torch.rand_like(joint_pos_info) * 0.02 - 0.01)
        joint_vel_noisy = joint_vel_info + (torch.rand_like(joint_vel_info) * 0.1 - 0.05)

        self.prev_velocity = self._robot.data.root_lin_vel_b.clone()

        # Ball linear velocity (x, y, z)
        ball_data = self.sphere_object.data.body_state_w[:, 0, [7, 8, 9]]

        # Optionally add noise to displacement
        noise_displace = (1.0 - self.action_noise) * estimated_displacement + \
            self.action_noise * torch.randn_like(estimated_displacement)

        obs_tensors = [
            self._robot.data.root_ang_vel_b if self.cfg.obs_ang_vel else None,
            self._robot.data.projected_gravity_b if self.cfg.obs_proj_grav else None,
            roll_noisy.unsqueeze(-1) if self.cfg.obs_roll else None,
            self.throwing_commands,
            joint_pos_noisy,
            joint_vel_noisy,
            self._actions,
            self.not_released_ball.unsqueeze(-1).float() if self.cfg.obs_notrelease else None,
            noise_displace.unsqueeze(-1) if self.cfg.obs_estimdisplace else None,
            estim_time.unsqueeze(-1) if self.cfg.obs_estimdisplace else None,
            ball_data if self.cfg.obs_estimdisplace else None,
        ]

        obs = torch.cat([t for t in obs_tensors if t is not None], dim=-1)
        obs = torch.clip(obs, -1000.0, 1000.0)

        if torch.sum(torch.isnan(obs)) > 0:
            print("NaNs in obs at dims:", torch.sum(torch.isnan(obs), dim=0))

        return {"policy": obs}


    def _get_rewards(self) -> torch.Tensor:
        # Cache indices
        self._cache_alpha_indices()

        # ------------
        # Throwing reward
        # ------------
        ball_pos = self.sphere_object.data.body_pos_w[:, 0, 0:3]
        right_hand_pos = self._robot.data.body_pos_w[:, self.right_hand_body_id, :]

        # Distance between ball and right hand
        dist = torch.norm(ball_pos - right_hand_pos, dim=1)
        throwing_reward_condition = (dist > 0.25) & (~self.throwing_reward_given)
        self.not_released_ball &= (dist <= 0.25)

        env_ids = torch.nonzero(throwing_reward_condition).reshape(-1)
        self.throwing_reward[:] = 0.0

        if len(env_ids) > 0:
            # check_ball_displacement returns (normalized_distance, landing_time)
            self.throwing_reward[env_ids], self.landing_time[env_ids] = self.check_ball_displacement(env_ids)
            self.throwing_reward[env_ids] = 1.0 - self.throwing_reward[env_ids]
            self.throwing_reward_given[env_ids] = True
            self.landing_time[env_ids] += self.episode_length_buf[env_ids] * self.step_dt

        ball_released_envs = env_ids

        # ------------
        # Collision / ball-not-thrown flags (for optional logging; stability reward removed)
        # ------------
        ball_not_thrown_cond = (torch.norm((right_hand_pos - ball_pos), dim=1) <= 0.25) & (self.reset_buf == 1)

        # Contact-based collision: ignore hand-ball contacts
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)
        mask = torch.where(
            torch.norm((right_hand_pos - ball_pos), dim=1) <= 0.25,
            torch.zeros(1, device=self.device),
            torch.ones(1, device=self.device),
        )
        if len(self.hand_ids) > 0:
            first_contact[:, self.hand_ids] &= mask.view(-1, 1).expand(-1, len(self.hand_ids)).to(torch.bool)
        collision = torch.sum(first_contact[:, self.non_feet_ids], dim=1) > 0

        # ------------
        # Action / torque / accel penalties
        # ------------
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

        rewards = {
            "throwing": torch.max(torch.zeros_like(self.throwing_reward), self.throwing_reward.clone())
            * self.cfg.throwing_reward_scale
            * self.max_episode_length_s,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
        }

        # ------------------
        # Optional reward terms
        # ------------------
        r_scale = 1.0

        # Base height reward (use hip_body_id)
        if getattr(self.cfg, "baseh_rew", False):
            base_height = self._robot.data.body_pos_w[:, self.hip_body_id, 2]
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

        # Energy reward uses torso+arm joints only
        if getattr(self.cfg, "energy_rew", False):
            num_arm_dofs = self._arm_dof_indices.shape[0]
            electricity_cost = torch.sum(
                torch.abs(
                    self._actions[:, :num_arm_dofs]
                    * self._robot.data.joint_vel[:, self._arm_dof_indices]
                ),
                dim=-1,
            )
            energy_vals_tensor = [[40, 70], [70, 100], [100, 130], [130, 160], [160, 200]]
            selected_bounds = energy_vals_tensor[self.energy_rew_vec[0].int()]
            min_energy, max_energy = selected_bounds[0], selected_bounds[1]
            reward_electricity = ((electricity_cost >= min_energy) & (electricity_cost <= max_energy)).float()
            reward_electricity *= self.step_dt * r_scale
            rewards["energy_rew"] = reward_electricity
            within = (electricity_cost >= min_energy) & (electricity_cost <= max_energy)
            den = torch.ones_like(within).sum()
            pct = 100.0 * within.float().sum() / den
            self.extras["log"]["energy_success_pct"] = pct

        # Ball release timing reward
        if getattr(self.cfg, "ballrel_rew", False):
            if len(ball_released_envs) == 0:
                reward_term_full = torch.zeros(self.num_envs, device=self.device)
                rewards["ballrel_rew"] = reward_term_full
                self.extras["log"]["ballrel_success_pct"] = torch.tensor(0.0, device=self.device)
            else:
                ball_released_time = self.episode_length_buf[ball_released_envs].float() * self.step_dt
                measured_val = ball_released_time
                bounds_list = [[0., 0.3], [0.3, 0.6], [0.6, 0.9], [0.9, 1.2], [1.2, 1.5]]
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

        # Body motion reward (hip velocity)
        if getattr(self.cfg, "bodymo_rew", False):
            if len(ball_released_envs) == 0:
                reward_term_full = torch.zeros(self.num_envs, device=self.device)
                rewards["bodymo_rew"] = reward_term_full
                self.extras["log"]["bodymo_success_pct"] = torch.tensor(0.0, device=self.device)
            else:
                measured_val = torch.norm(
                    self._robot.data.body_vel_w[:, self.hip_body_id, :3],
                    dim=-1,
                )
                bounds_list = [[0., 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
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

        # Left arm posture (left hand height)
        if getattr(self.cfg, "lftarm_rew", False):
            measured_val = self._robot.data.body_pos_w[:, self.left_hand_body_id, 2]
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

        # Right arm at release (right hand height)
        if getattr(self.cfg, "rgtarmrel_rew", False):
            if len(ball_released_envs) == 0:
                reward_term_full = torch.zeros(self.num_envs, device=self.device)
                rewards["rgtarmrel_rew"] = reward_term_full
                self.extras["log"]["rgtarmrel_success_pct"] = torch.tensor(0.0, device=self.device)
            else:
                measured_val = self._robot.data.body_pos_w[:, self.right_hand_body_id, 2]
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

        # Accumulate episode sums
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return torch.stack(list(rewards.values())).T


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Reset robot + base DirectRLEnv state
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Cache indices
        self._cache_alpha_indices()

        # Arm-only flag (same behavior as before)
        if self.cfg.arm_only:
            self.body_actions = 0.0
        else:
            self.body_actions = 1.0

        if self.cfg.distance_throw:
            self.distance_range = [max(self.distance_range[0], 4.0), max(self.distance_range[1], 4.0)]

        if self.cfg.no_proj_motion:
            self.cfg.obs_estimdisplace = False

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Sample new throwing commands and targets
        self._sample_throwing_commands(env_ids)
        self.target_positions[env_ids] = self.calculate_target_offset(env_ids)

        # Reset throwing-related flags
        self.throwing_reward[env_ids] = -1.0
        self.landing_time[env_ids] = -1.0
        self.throwing_reward_given[env_ids] = False
        self.not_released_ball[env_ids] = True
        self.sum_open_hand_action[env_ids] = 0.0
        self.released_ball_t[env_ids] = -1.0

        # -------------------------
        # Reset robot state (Alpha)
        # -------------------------
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]

        # Add some noise to arm joints around default
        arm_noise = torch.zeros_like(joint_pos[:, self._arm_dof_indices]).uniform_(
            -self.cfg.arm_dr_range,
            self.cfg.arm_dr_range,
        )
        joint_pos[:, self._arm_dof_indices] += arm_noise

        # Small global joint noise
        joint_pos += torch.zeros_like(joint_pos).uniform_(-0.05, 0.05)

        # Yaw offset for robot root
        if self.robot_yaw_offset_rad != 0.0:
            offset_w = math.cos(self.robot_yaw_offset_rad * 0.5)
            offset_z = math.sin(self.robot_yaw_offset_rad * 0.5)
            offset_quat = torch.zeros_like(default_root_state[:, 3:7])
            offset_quat[:, 0] = offset_w
            offset_quat[:, 3] = offset_z
            default_root_state[:, 3:7] = self._quat_multiply(offset_quat, default_root_state[:, 3:7])

        # Shift by terrain origins
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # Write state to sim
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # reset stability flags
        self.stability_penalty_this_ep[env_ids] = False

        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(env_ids)

    def calculate_target_offset(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Compute target position in *local env frame* from throwing commands.

        Returns a (len(env_ids), 3) tensor: [x, y, z] offset from env origin.
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return torch.zeros((0, 3), device=self.device)

        cmds = self.throwing_commands[env_ids]
        distances = cmds[:, 0]
        headings = cmds[:, 1] + self.target_heading_offset_rad + self.robot_yaw_offset_rad

        # Simple flat ground target at ~0.5m height (tune later if needed)
        x = distances * torch.cos(headings)
        y = distances * torch.sin(headings)
        z = torch.full_like(distances, 0.5)  # target height above ground

        return torch.stack([x, y, z], dim=-1)

@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward