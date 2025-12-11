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

        self.reward_components = len(self._episode_sums.keys())
        self.reward_component_names = list(self._episode_sums.keys())
        self.reward_component_task_rew = ["throwing"]

        if self.cfg.baseh_rew:
            self.reward_component_names += ["baseh_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["baseh_rew"]
            self._episode_sums["baseh_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if self.cfg.energy_rew:
            self.reward_component_names += ["energy_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["energy_rew"]
            self._episode_sums["energy_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if self.cfg.ballrel_rew:
            self.reward_component_names += ["ballrel_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["ballrel_rew"]
            self._episode_sums["ballrel_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if self.cfg.bodymo_rew:
            self.reward_component_names += ["bodymo_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["bodymo_rew"]
            self._episode_sums["bodymo_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if self.cfg.lftarm_rew:
            self.reward_component_names += ["lftarm_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["lftarm_rew"]
            self._episode_sums["lftarm_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if self.cfg.rgtarmrel_rew:
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
        self.actions = actions.clone()
        default_positions = self._robot.data.default_joint_pos[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]].clone() # non arm indices
        scalar_factor = 3
        self._processed_actions = self.cfg.action_scale * self._actions[:,:-1] + default_positions


        ##########  TODO: Need to change finger actions based on alpha hand design  ##########
        # Define finger actions
        finger_positions = torch.tensor([-0.35, -0.61, 0, 0.61, 0.35, 0.61, 0.35], device=self.device) * scalar_factor
        finger_positions[1] /= 2  # right two link
        finger_positions[0] /= 2  # right one link

        # Create full finger action array from -14 to -1 (14 positions)
        full_finger_actions = torch.zeros((self.num_envs, 14), device=self.device)

        # Map finger positions to their indices (relative to the 14-position array)
        # Index mapping: -14=0, -13=1, ..., -2=12, -1=13
        finger_indices = [-3, -1, -9, -4, -10, -5, -11]  # Original indices
        array_indices = [14 + idx for idx in finger_indices]  # Convert to 0-based indices: [11, 13, 5, 10, 4, 9, 3]

        # Set finger positions at their corresponding array positions
        for i, pos_idx in enumerate(array_indices):
            full_finger_actions[:, pos_idx] = finger_positions[i]

        # Set finger actions to 0 if hand is open (using smoothed actions)
        hand_open_mask = (self._actions[:, -1] >= 0)
        full_finger_actions[~hand_open_mask] = 0
        #######################################################################################

        # Append to _processed_actions
        self._processed_actions = torch.cat([self._processed_actions, full_finger_actions], dim=1)

        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(torch.arange(self.num_envs,device=self.device))

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._processed_actions)#[:,:-1])

    def _get_observations(self) -> dict:
        ##########  TODO: Need to change on alpha  ##########
        if not hasattr(self, "non_feet_ids"):
            self.non_feet_ids,_ = self._contact_sensor.find_bodies("^(?!.*ankle).*$")#spot:  "^(?!.*lleg).*$" , chuck: "^(?!.*foot).*$", g1: ankle
            self.hand_ids,_ = self._contact_sensor.find_bodies(".*(palm|five|six|three|four|zero|one|two).*")# spot:  ".*(fngr|wr1).*", chuck:  ".*(finger|cylinder).*", g1: .*(palm|five|six|three|four|zero|one|two).*
            self._hip_ids,_ = self._contact_sensor.find_bodies(".*(hip).*")
            self.left_hand_ids,_ = self._contact_sensor.find_bodies(".*left_(palm).*")# spot:  ".*(fngr|wr1).*", chuck:  ".*(finger|cylinder).*", g1: .*(palm|five|six|three|four|zero|one|two).*
            self.right_hand_ids,_ = self._contact_sensor.find_bodies(".*right_(palm).*")# spot:  ".*(fngr|wr1).*", chuck:  ".*(finger|cylinder).*", g1: .*(palm|five|six|three|four|zero|one|two).*

        self._previous_actions = self._actions.clone()
        # remove hand observations
        # non hand indices : [:23]
        joint_pos_info = (self._robot.data.joint_pos - self._robot.data.default_joint_pos)[:,:23]
        joint_vel_info = self._robot.data.joint_vel[:,:23]
        
        estimated_displacement,estim_time = self.check_ball_displacement(torch.arange(self.num_envs,device=self.device))
        estimated_displacement = 1 - estimated_displacement
        #base_height = self._robot.data.root_pos_w[:, 2]-self.min_base_height
        w,x,y,z = self._robot.data.root_quat_w.T
        roll = torch.atan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x**2 + y**2)).to(self.device)

        noise_displace = (1 - self.action_noise) * estimated_displacement+ self.action_noise * torch.randn_like(estimated_displacement)
        
        self.prev_velocity = self._robot.data.root_lin_vel_b.clone()

        ball_data = self.sphere_object.data.body_state_w[:, 0, [7,8,9]] # z vel

        obs = torch.cat( 
            [
                tensor
                for tensor in (

                    self._robot.data.root_ang_vel_b if self.cfg.obs_ang_vel else None,
                    self._robot.data.projected_gravity_b if self.cfg.obs_proj_grav else None,
                    (roll.float() + (torch.rand_like(roll) * 0.02 - 0.01)).unsqueeze(-1) if self.cfg.obs_roll else None,
                    self.throwing_commands,#
                    joint_pos_info + (torch.rand_like(joint_pos_info) * 0.02 - 0.01),
                    joint_vel_info + (torch.rand_like(joint_vel_info) * 0.1 - 0.05),
                    self._actions,
                    self.not_released_ball.unsqueeze(-1).float() if self.cfg.obs_notrelease else None,
                    noise_displace.unsqueeze(-1) if self.cfg.obs_estimdisplace else None,#estimated_displacement.unsqueeze(-1).float() if self.cfg.obs_estimdisplace else None,
                    estim_time.unsqueeze(-1) if self.cfg.obs_estimdisplace else None,
                    ball_data if self.cfg.obs_estimdisplace else None,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        obs = torch.clip(obs, -1000,1000)
        #assert False
        observations = {"policy": obs}
        
        if torch.sum(torch.isnan(obs)) > 0:
            print(torch.sum(torch.isnan(obs),dim=0))
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # throwing reward:
        # measure euclidean distance between ball and hand 
        if self.cfg.no_proj_motion:
            ##########  TODO: Need to change on alpha  ##########
            dist = torch.norm(self.sphere_object.data.body_pos_w[:, 0, 0:3] - self._robot.data.body_pos_w[:, [-15], :].squeeze(1), dim=1) 
            throwing_reward_condition = (dist > 0.25)

            target_positions = self.target_positions[:] + self._terrain.env_origins[:]
            targ_dist = torch.norm(self.sphere_object.data.body_pos_w[:, 0, 0:3] - target_positions, dim=1).reshape(self.num_envs)
            on_ground = self.sphere_object.data.body_pos_w[:, 0, 2] < 0.05

            # set to targ_dist if ball distances -1 or if targ_distance smaller than the current amount (excluding -1)
            self.ball_distances[:] = torch.where(~on_ground & ((self.ball_distances[:] == -1) | (targ_dist < self.ball_distances[:])), targ_dist, self.ball_distances[:])
            env_ids = torch.nonzero((self.reset_buf == 1) & throwing_reward_condition).reshape(-1)
            self.throwing_reward[:] = 0
            self.throwing_reward[env_ids] = self.ball_distances[env_ids]/self.throwing_commands[env_ids,0]
            self.throwing_reward[env_ids] = 1 - torch.min(torch.tensor(1.), self.throwing_reward[env_ids])
            assert False
        else:
            dist = torch.norm(self.sphere_object.data.body_pos_w[:, 0, 0:3] - self._robot.data.body_pos_w[:, [-15], :].squeeze(1), dim=1) 
            throwing_reward_condition = (dist > 0.25) & (~self.throwing_reward_given)
            self.not_released_ball &= (dist <= 0.25)
            env_ids = torch.nonzero(throwing_reward_condition).reshape(-1)
            self.throwing_reward[:] = 0
            if len(env_ids) > 0: 
                self.throwing_reward[env_ids], self.landing_time[env_ids] = self.check_ball_displacement(env_ids)
                self.throwing_reward[env_ids] = 1 - self.throwing_reward[env_ids]
                self.throwing_reward_given[env_ids] = True
                self.landing_time[env_ids] += self.episode_length_buf[env_ids] * self.step_dt
                ball_data = self.sphere_object.data.body_state_w[env_ids, 0, 9] # z vel
        ball_released_envs = env_ids

        # roll reward
        # w,x,y,z = self._robot.data.root_quat_w.T
        # roll = torch.atan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x**2 + y**2))
        # make sure roll not nan
        # roll = torch.where(torch.isnan(roll), torch.zeros_like(roll), roll)
        # roll_rew = (-1/(1+torch.exp(-10*(torch.abs(roll)-0.3))))*(1-torch.exp(-(torch.abs(roll)-0.1)/0.1))
        # stability reward
        # base_height_cond = (self._robot.data.root_pos_w[:, 2] <= self.min_base_height) # 0.42 for anymal
        
        ##########  TODO: Need to change on alpha  ##########
        hand_positions = self._robot.data.body_pos_w[:, [-1], :].reshape(self.num_envs,3) # 22 for anymal
        ball_positions = self.sphere_object.data.root_pos_w.clone()
        ball_not_thrown_cond = (torch.norm((hand_positions-ball_positions),dim=1) <= 0.25) & (self.reset_buf == 1) 
        
        #bad_throw_cond = (self.throwing_reward > -1) & (self.throwing_reward < 0.05)

        # collision detection
        # "finger" or "cylinder"
        # include all ids except feet
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)#[:,non_feet_ids]
        mask = torch.where(torch.norm((hand_positions-ball_positions),dim=1) <= 0.25, torch.zeros(1,device=self.device),torch.ones(1,device=self.device))

        first_contact[:,self.hand_ids] &= mask.view(-1,1).expand(-1, len(self.hand_ids)).to(torch.bool) # filter out collisions with ball touching the hand
        collision = torch.sum(first_contact[:,self.non_feet_ids],dim=1) > 0

        self.stability_penalty_this_ep |= (ball_not_thrown_cond | collision) 
        stability_rew = ((self.reset_buf == 1) & (~self.stability_penalty_this_ep)).float()
        # if self.cfg.nonsparse_stability_reward:
            # stability_rew = ((~(base_height_cond | ball_not_thrown_cond | collision)).float() / self.max_episode_length_s) * self.step_dt

        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

        rewards = {
            "throwing": torch.max(torch.zeros_like(self.throwing_reward),self.throwing_reward.clone()) * self.cfg.throwing_reward_scale * self.max_episode_length_s,
            # "roll": roll_rew * self.cfg.roll_reward_scale * self.step_dt,
            # "stability": stability_rew * self.cfg.stability_reward_scale * self.max_episode_length_s,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
        }

        if self.cfg.baseh_rew:
            ##########  TODO: Need to change on alpha  ##########
            base_height = self._robot.data.body_pos_w[:, self._hip_ids[0], 2] # 0.99 - 1.4 (jumping high)
            bounds_list = [[0.35,0.4],[0.4,0.45],[0.45,0.5],[0.5,0.55],[0.55,0.6]] # starts around 0.6
            selected_bounds = bounds_list[self.baseh_rew_vec[0].int()]  # (num_envs, 2)
            min_value, max_value = selected_bounds[0], selected_bounds[1]
            reward_term = ((base_height >= min_value) & (base_height <= max_value)).float()
            reward_term *= self.step_dt
            rewards["baseh_rew"] = reward_term
            within  = (base_height >= min_value) & (base_height <= max_value)
            den = torch.ones_like(within).sum()                        # total valid feet this step
            pct = 100.0 * within.float().sum() / den                      # scalar %
            self.extras["log"]["base_height_success_pct"] = pct

        r_scale = 1.0
        if self.cfg.energy_rew:
            ##########  TODO: Need to change on alpha  ##########
            electricity_cost = torch.sum(
                torch.abs(self._actions[:,:-1] * self._robot.data.joint_vel[:,:23]),
                dim=-1,
            )
            energy_vals_tensor = [[40,70],[70,100],[100,130],[130,160],[160,200]] # 26 -> 140
            selected_bounds = energy_vals_tensor[self.energy_rew_vec[0].int()]  # (num_envs, 2)
            min_energy = selected_bounds[0]
            max_energy = selected_bounds[1]
            reward_electricity = ((electricity_cost >= min_energy) & (electricity_cost <= max_energy)).float()
            reward_electricity *= self.step_dt * r_scale
            rewards["energy_rew"] = reward_electricity
            within  = (electricity_cost >= min_energy) & (electricity_cost <= max_energy)
            den = torch.ones_like(within).sum()                        # total valid feet this step
            pct = 100.0 * within.float().sum() / den                      # scalar %
            self.extras["log"]["energy_success_pct"] = pct

        if self.cfg.ballrel_rew:
            # episode_length_buf is in steps; multiply by step_dt (s/step) to get seconds
            if len(ball_released_envs) == 0:
                reward_term = torch.zeros(self.num_envs, device=self.device)
                rewards["ballrel_rew"] = reward_term
                self.extras["log"]["ballrel_success_pct"] = torch.tensor(0., device=self.device)
            else:
                ball_released_time = self.episode_length_buf[ball_released_envs].float() * self.step_dt
                measured_val = ball_released_time # around 0.2
                bounds_list = [[0.,0.3],[0.3,0.6],[0.6,0.9],[0.9,1.2],[1.2,1.5]] # starts around 0.6
                selected_bounds = bounds_list[self.ballrel_rew_vec[0].int()]  # (num_envs, 2)
                min_value, max_value = selected_bounds[0], selected_bounds[1]
                reward_term = ((measured_val >= min_value) & (measured_val <= max_value)).float()
                reward_term *= r_scale
                reward_term_full = torch.zeros(self.num_envs, device=self.device)
                reward_term_full[ball_released_envs] = reward_term
                rewards["ballrel_rew"] = reward_term_full
                within  = (measured_val >= min_value) & (measured_val <= max_value)
                den = torch.ones_like(within).sum()                        # total valid feet this step
                pct = 100.0 * within.float().sum() / den                      # scalar %
                self.extras["log"]["ballrel_success_pct"] = pct

        if self.cfg.bodymo_rew:
            if len(ball_released_envs) == 0:
                reward_term = torch.zeros(self.num_envs, device=self.device)
                rewards["bodymo_rew"] = reward_term
                self.extras["log"]["bodymo_success_pct"] = torch.tensor(0., device=self.device)
            else:
                ##########  TODO: Need to change on alpha  ##########
                measured_val = torch.norm(self._robot.data.body_vel_w[:, self._hip_ids[0], :3], dim=-1) # around 2
                bounds_list = [[0.,1.0],[1.0,2],[2,3],[3,4],[4,5]] # starts around 0.6
                selected_bounds = bounds_list[self.bodymo_rew_vec[0].int()]  # (num_envs, 2)
                min_value, max_value = selected_bounds[0], selected_bounds[1]
                reward_term = ((measured_val >= min_value) & (measured_val <= max_value)).float()
                reward_term *= r_scale
                reward_term_full = torch.zeros(self.num_envs, device=self.device)
                reward_term_full[ball_released_envs] = reward_term[ball_released_envs]
                rewards["bodymo_rew"] = reward_term_full
                within  = (measured_val >= min_value) & (measured_val <= max_value)
                den = torch.ones_like(within).sum()                        # total valid feet this step
                pct = 100.0 * within.float().sum() / den                      # scalar %
                self.extras["log"]["bodymo_success_pct"] = pct

        if self.cfg.lftarm_rew:
            ##########  TODO: Need to change on alpha  ##########
            measured_val = self._robot.data.body_pos_w[:, self.left_hand_ids[0], 2] # around 0.8
            bounds_list = [[0.6,0.8],[0.8,1.0],[1.0,1.2],[1.2,1.4],[1.4,1.6]] # starts around 0.6
            selected_bounds = bounds_list[self.lftarm_rew_vec[0].int()]  # (num_envs, 2)
            min_value, max_value = selected_bounds[0], selected_bounds[1]
            reward_term = ((measured_val >= min_value) & (measured_val <= max_value)).float()
            reward_term *= self.step_dt * r_scale
            rewards["lftarm_rew"] = reward_term
            within  = (measured_val >= min_value) & (measured_val <= max_value)
            den = torch.ones_like(within).sum()                        # total valid feet this step
            pct = 100.0 * within.float().sum() / den                      # scalar %
            self.extras["log"]["lftarm_success_pct"] = pct
        if self.cfg.rgtarmrel_rew:
            if len(ball_released_envs) == 0:
                reward_term = torch.zeros(self.num_envs, device=self.device)
                rewards["rgtarmrel_rew"] = reward_term
                self.extras["log"]["rgtarmrel_success_pct"] = torch.tensor(0., device=self.device)
            else:
                ##########  TODO: Need to change on alpha  ##########
                measured_val = self._robot.data.body_pos_w[:, self.right_hand_ids[0], 2] # around 0.5
                bounds_list = [[0.3,0.4],[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.7,0.8]] # starts around 0.6
                selected_bounds = bounds_list[self.rgtarmrel_rew_vec[0].int()]  # (num_envs, 2)
                min_value, max_value = selected_bounds[0], selected_bounds[1]
                reward_term = ((measured_val >= min_value) & (measured_val <= max_value)).float()
                reward_term *= r_scale
                reward_term_full = torch.zeros(self.num_envs, device=self.device)
                reward_term_full[ball_released_envs] = reward_term[ball_released_envs]
                rewards["rgtarmrel_rew"] = reward_term_full
                within  = (measured_val >= min_value) & (measured_val <= max_value)
                den = torch.ones_like(within).sum()                        # total valid feet this step
                pct = 100.0 * within.float().sum() / den                      # scalar %
                self.extras["log"]["rgtarmrel_success_pct"] = pct

        #reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
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
        
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        

        if self.cfg.arm_only:
            self.body_actions = 0.
        else:
            self.body_actions = 1. # max is 100%

        if self.cfg.distance_throw:
            self.distance_range = [max(self.distance_range[0], 4), max(self.distance_range[1], 4)]
        
        if self.cfg.no_proj_motion:
            self.cfg.obs_estimdisplace = False



        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        self._sample_throwing_commands(env_ids)

        self.target_positions[env_ids] = self.calculate_target_offset(env_ids)
        self.throwing_reward[env_ids] = -1 
        self.landing_time[env_ids] = -1
        self.throwing_reward_given[env_ids] = False
        self.not_released_ball[env_ids] = True
        self.sum_open_hand_action[env_ids] = 0
        self.released_ball_t[env_ids] = -1
        # self.stability_penalty_this_ep[env_ids] = False # removed on purpose


        ##### TODO: Need to change on alpha  ##########
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] 
        #joint_pos[:,[1,3,5, 7,9, 11]] += torch.zeros_like(joint_pos[:,[1,3,5, 7,9, 11]]).uniform_(-self.cfg.arm_dr_range, self.cfg.arm_dr_range) # hand .uniform_(-1,1)
        joint_pos[:,[5,6,9,10,13,14,17,18,21,22]] += torch.zeros_like(joint_pos[:,[5,6,9,10,13,14,17,18,21,22]]).uniform_(-0.3, 0.3) # hand .uniform_(-1,1)
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
        ##############################################################

        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(env_ids)


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