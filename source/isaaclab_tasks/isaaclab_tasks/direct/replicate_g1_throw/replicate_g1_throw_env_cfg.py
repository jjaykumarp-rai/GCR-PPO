# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the Replicate G1 Throw (DirectRLEnv) task.

This module defines the *configuration objects* used by IsaacLab to construct:
- Simulation settings (time-step, physics materials, solver iterations, etc.)
- Terrain (plane or generator-based terrain)
- Scene (number of env instances, spacing, replication)
- Assets (robot articulation, sphere/ball, target marker)
- Events / randomizations (domain randomization hooks via EventTerm)
- Sensors (contact sensor)
- Task-specific hyperparameters (reward scales, observation toggles, reset noise, target sampling)

KT notes:
- This file is configuration-only (dataclass-style). The actual environment logic
  (step(), rewards, reset, observations) lives in the corresponding env file.
- `@configclass` is IsaacLab's pattern for structured configs that can be composed
  and overridden by Hydra/CLI.
"""

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # Example import (not used here)

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

###### part of CDR-1630 ######
# Terrain generation utilities (optional; currently plane terrain is used by default)
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg

# Event / domain randomization utilities
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

# Asset + sensor configs used by the environment
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns  # RayCasterCfg/patterns imported but not used here
from .alpha_utils import ALPHA_CFG


# -----------------------------------------------------------------------------
# Optional terrain generator configuration (commented out in the main cfg below).
# This describes a tiled terrain with multiple sub-terrain types.
# -----------------------------------------------------------------------------
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(5.0, 5.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    sub_terrains={
        # Flat plane tiles
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        # Random heightfield roughness
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
        # Sloped pyramids
        "slopey": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.05, 0.15), platform_width=0.0
        ),
        # Inverted sloped pyramids (valley-like)
        "slopey_inverted": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.05, 0.15), platform_width=0.0, inverted=True
        ),
    },
)


@configclass
class EventCfg:
    """Event (domain randomization) configuration.

    Event terms are hooks that IsaacLab can execute at specified times:
    - "startup": once at environment creation/reset
    - "reset": at every episode reset
    - "interval": periodically during rollouts

    Here we enable rigid body material randomization to improve robustness.
    """

    # -------------------------------------------------------------------------
    # Physics material randomization for the robot.
    #
    # This randomizes physical material properties (friction/restitution) of
    # bodies matching the regex `.*` under the "robot" scene entity.
    # -------------------------------------------------------------------------
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            # Apply to all robot rigid bodies
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # Keep fixed ranges here (same min=max) but retain buckets for future flexibility
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # -------------------------------------------------------------------------
    # The blocks below are intentionally commented out (kept for reference).
    # They represent experiments for joint friction randomization and base mass
    # perturbations (useful in domain randomization for sim2real).
    # -------------------------------------------------------------------------

    # arm_joints = [
    #     # Left arm joints
    #     "left_shoulder_pitch_joint",
    #     "left_shoulder_roll_joint",
    #     "left_shoulder_yaw_joint",
    #     "left_elbow_joint",
    #     "left_wrist_roll_joint",
    #
    #     # Right arm joints
    #     "right_shoulder_pitch_joint",
    #     "right_shoulder_roll_joint",
    #     "right_shoulder_yaw_joint",
    #     "right_elbow_joint",
    #     "right_wrist_roll_joint",
    # ]

    # arm_friction_randomization = EventTerm(
    #     func=mdp.randomize_joints_friction,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=arm_joints),
    #         "friction_distribution_params": (0.7, 1.3),  # 30% up/down variation
    #         "operation": "scale",
    #     },
    # )

    # Why 0.7 → 1.3?
    # Hardware joints often have 20–40% real-world friction variability.
    # Too high variation breaks training; too low variation doesn’t matter.
    # This range is a practical sweet spot for throwing / fast arm motions.

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="PLINTH"),  # base link name for this robot
    #         "mass_distribution_params": (-0.5, 0.5),
    #         "operation": "add",
    #     },
    # )

    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(0.0, 4.0),
    #     params={"velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}},
    # )


@configclass
class ReplicateG1ThrowEnvCfg(DirectRLEnvCfg):
    """Top-level configuration for the Replicate G1 Throw DirectRLEnv.

    This config class is consumed by the environment implementation to set up:
    - Episode timing and action/observation sizes
    - Simulation properties (dt, rendering rate, physics materials)
    - Terrain and assets
    - Sensors and event randomizations
    - Observation toggles and reward scale hyperparameters
    - Target generation / curriculum parameters

    Any attribute defined here is typically accessible in the env as `self.cfg.<...>`.
    """

    # -------------------------------------------------------------------------
    # Environment roll-out timing and spaces
    # -------------------------------------------------------------------------
    episode_length_s = 2.0
    decimation = 4  # action repeats / control decimation relative to sim dt
    action_scale = 0.5

    # Actions: [ torso (1), right arm (7), left arm (7), grip (1) ] = 16
    action_space = 16

    # Observation vector dimension expected by the policy
    observation_space = 105

    # No separate state for asymmetric actor-critic (set to 0)
    state_space = 0

    # Whether to include air resistance in dynamics (handled in env logic)
    air_resistance = False

    # -------------------------------------------------------------------------
    # Simulation configuration
    # -------------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,  # physics timestep
        render_interval=decimation,  # render at control rate
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # Combine rules determine how friction/restitution are composed on contact
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # -------------------------------------------------------------------------
    # Terrain configuration
    #
    # Two options exist here:
    # 1) Generated multi-terrain (commented out)
    # 2) Simple plane terrain (enabled)
    # -------------------------------------------------------------------------
    '''
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.1, 0.1),
        ),
        debug_vis=True,
    )
    '''

    # Default terrain: infinite plane with a visual material
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.1, 0.1),
        ),
        debug_vis=False,
    )

    # -------------------------------------------------------------------------
    # Ball (sphere) configuration
    # -------------------------------------------------------------------------
    sphere_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.023,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                rigid_body_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.085),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    # -------------------------------------------------------------------------
    # Target configuration (visual/kinematic marker the robot should aim for)
    # -------------------------------------------------------------------------
    target_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target",
        spawn=sim_utils.CylinderCfg(
            radius=0.5,   # disk radius (m)
            height=0.03,  # disk thickness (m)
            axis="X",     # cylinder axis along X => face normal aligns with +X/-X
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # target is positioned by the env, not physics
                rigid_body_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # bright red
            ),
        ),
        # init_state can be specified here; currently left to env logic
    )

    # -------------------------------------------------------------------------
    # Scene configuration
    # -------------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # -------------------------------------------------------------------------
    # Ball tracking debug helpers
    #
    # When enabled, the env draws debug lines following each ball (or a configured subset of envs).
    # Colors default to the sphere preview color so the trace matches the visual ball.
    # This relies on `omni.debugdraw` and is only active when rendering is enabled.
    # -------------------------------------------------------------------------
    ball_trace_enabled = True
    ball_trace_env_ids: list[int] | None = None
    ball_trace_history_length = 200
    ball_trace_color: tuple[float, float, float, float] | None = None
    ball_trace_thickness = 4.0

    # -------------------------------------------------------------------------
    # Event configuration (domain randomization hooks)
    # -------------------------------------------------------------------------
    events: EventCfg = EventCfg()

    # -------------------------------------------------------------------------
    # Sensors
    # -------------------------------------------------------------------------
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # -------------------------------------------------------------------------
    # Robot configuration
    # -------------------------------------------------------------------------
    # Start from shared ALPHA_CFG and override the prim path for this environment.
    robot: ArticulationCfg = ALPHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # -------------------------------------------------------------------------
    # Reward scales / penalties
    #
    # Convention used in many IsaacLab tasks:
    # - Positive values: rewards (maximize)
    # - Negative values: penalties (minimize)
    # -------------------------------------------------------------------------
    throwing_reward_scale = 3.0            # prioritize target accuracy
    throw_height_reward_scale = 1.0        # secondary shaping term
    throw_height_target = 0.6              # desired apex/height metric target
    action_rate_reward_scale = -1e-3       # penalize rapid action changes
    joint_torque_reward_scale = -2.5e-6    # penalize high torques
    joint_vel_reward_scale = -1e-4         # penalize high joint velocities
    joint_accel_reward_scale = -2.5e-8     # penalize high joint accelerations
    joint_vel_penalty_clip = 1.0e4         # clip for stability in penalty terms
    joint_accel_penalty_clip = 1.0e4
    action_limit_penalty_scale = -1e-3     # penalize exceeding action limits

    # throw_time_reward_scale = 1.0
    # zvel_reward_scale = 0.75

    # -------------------------------------------------------------------------
    # Reset and observation randomization (initial state noise)
    # -------------------------------------------------------------------------
    right_arm_init_range = 0.35
    other_joint_init_range = 0.03
    joint_pos_noise_range = (-0.05, 0.05)
    joint_vel_noise_range = (-0.07, 0.07)

    # -------------------------------------------------------------------------
    # Observation toggles
    #
    # These booleans typically gate which terms are concatenated into the
    # observation vector by the environment implementation.
    # -------------------------------------------------------------------------
    obs_lin_vel = True
    obs_ang_vel = False
    obs_proj_grav = True

    obs_baseheight = False
    obs_footangle = False
    obs_notrelease = True
    obs_estimdisplace = True
    obs_torque = False
    obs_ball_state = True
    obs_hand_pose = True
    obs_time = True
    obs_target_rel = True
    obs_processed_actions = True

    # Threshold used in throwing reward/logic (exact usage lives in env code)
    r_throw_thresh = 0.5

    # -------------------------------------------------------------------------
    # Target placement controls
    # -------------------------------------------------------------------------
    min_throw_dist = 2.5
    target_fov_deg = 30.0
    target_height_range = (0.1, 1.0)

    # Rotate target heading relative to the world forward (+X).
    # Comment indicates Alpha faces +Y, so offset is used to align frames.
    target_heading_offset_deg = 90.0
    robot_yaw_offset_deg = 0.0

    # -------------------------------------------------------------------------
    # Experimental flags / ablations
    # -------------------------------------------------------------------------
    use_stability = True                  # stability reward enabled (comment suggests may not be needed)
    no_proj_motion = False
    nonsparse_stability_reward = False    # stability reward only contains collision + ball-not-thrown terms
    max_throw_dist = 4.0

    # -------------------------------------------------------------------------
    # Presentation helpers
    # -------------------------------------------------------------------------
    sequential_distance_mode = True
    sequential_distance_start = 2.5
    sequential_distance_step = 0.1

    # ---------------------------------------------------------------------
    # Migration notes: changes for moving from `throwing` → `replicate_g1_throw`
    # ---------------------------------------------------------------------
    obs_roll = False

    distance_throw = False
    arm_only = False

    # Which hand throws (affects target placement side and grasp link choice)
    throw_hand_side = "right"

    # Additional stability-related reward knobs
    roll_reward_scale = 0.0
    stability_reward_scale = 0.25  # was experimented at smaller values (see comment)
    r_stability_thresh = 0.22
    # ---------------------------------------------------------------------
