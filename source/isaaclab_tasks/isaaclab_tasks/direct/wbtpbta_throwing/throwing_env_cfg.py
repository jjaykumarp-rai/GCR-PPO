# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import G1_CFG  # isort: skip
import isaaclab.terrains as terrain_gen
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


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
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
        "slopey": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.05, 0.15), platform_width=0.0
        ),
        "slopey_inverted": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.05, 0.15), platform_width=0.0, inverted=True
        ),
    },
)


@configclass
class EventCfg:
    """Configuration for randomization and perturbations."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # base_link for chuck, base for spot, imu_link for G1
            "asset_cfg": SceneEntityCfg("robot", body_names="imu_link"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )
    # Example push disturbance if needed later
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(0.0, 4.0),
    #     params={"velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}},
    # )


@configclass
class CommandsCfg:
    """Command configuration for throwing.

    This mirrors the structure used in Ma et al.:
      * resampling_time is the duration of a single throw command
        (target EE velocity / acceleration) in seconds.
      * The reward terms we imported use this to compute how many
        steps per command, and to window the 'throw phase' at the end.
    """

    # Duration of one command/throw target [s]
    resampling_time: float = 0.4

    # You can add more fields later (e.g., min/max distance, curriculum, etc.)


@configclass
class RewardsCfg:
    """Reward configuration.

    This holds all parameters needed by the Ma et al. reward terms,
    in particular:

      * pos_reward_duration: how long (in seconds) before the end of
        each command we reward EE tracking.
      * vel_*_sigma: shaping parameters for the smooth
        1 / (1 + error / sigma) rewards.

    The weights can be used when you build the final reward dict.
    """

    # Duration of the 'tracking window' at the end of each command [s]
    pos_reward_duration: float = 0.08  # e.g. 80 ms like in the paper (tune as needed)

    # Shaping parameters for reward terms
    vel_tracking_sigma: float = 2.0
    fingers_not_blocking_sigma: float = 0.1

    # Weights for Ma-style terms (you'll use these when summing reward)
    vel_tracking_weight: float = 4000.0
    fingers_not_blocking_weight: float = 1000.0

    # Legacy / additional task weights (keep from your original config)
    throwing_reward_scale: float = 2.05
    roll_reward_scale: float = 0.43
    stability_reward_scale: float = 0.25
    throw_height_reward_scale: float = 2.0
    action_rate_reward_scale: float = -1e-3
    joint_torque_reward_scale: float = -2.5e-6
    joint_accel_reward_scale: float = -2.5e-8
    # r_throw_thresh, r_stability_thresh etc. are kept in the main cfg for now.


@configclass
class ThrowingGeneralEnvCfg(DirectRLEnvCfg):
    """Direct RL environment configuration for G1 whole-body throwing."""

    # ------------------------------------------------------------------
    # Env / RL interface
    # ------------------------------------------------------------------

    # Episode duration (in seconds) used for reward normalization
    episode_length_s = 2.0

    # Physics decimation: env.step() every `decimation` sim steps.
    # With dt = 1/400 and decimation = 4, we get 100 Hz control.
    decimation = 4

    action_scale = 0.5
    action_space = 24
    observation_space = 100
    state_space = 0

    # Optional drag model for the ball (Ma et al. use a light ball with drag).
    air_resistance = False

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 400,  # 400 Hz physics to allow 100 Hz nominal + 400 Hz residual loop
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------
    # For now, a flat plane. You can re-enable the generator terrain if needed.
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
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
                     f"TilesMarbleSpiderWhiteBrickBondHoned/"
                     f"TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.1, 0.1),
        ),
        debug_vis=False,
    )

    # ------------------------------------------------------------------
    # Objects: ball and target
    # ------------------------------------------------------------------
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

    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target",
        spawn=sim_utils.CylinderCfg(
            radius=0.15,  # disk radius (m)
            height=0.01,  # very thin -> disk-like
            axis="X",     # axis along X => faces normal to X
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                rigid_body_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # bright red
            ),
        ),
        # init_state can be set dynamically per-env when sampling targets
        # init_state=RigidObjectCfg.InitialStateCfg(
        #     pos=(4.0, 0.0, 1.0),
        #     rot=(1.0, 0.0, 0.0, 0.0),
        # ),
    )

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # ------------------------------------------------------------------
    # Events / randomization
    # ------------------------------------------------------------------
    events: EventCfg = EventCfg()

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # ------------------------------------------------------------------
    # Robot
    # ------------------------------------------------------------------
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ------------------------------------------------------------------
    # Ma-style commands + rewards
    # ------------------------------------------------------------------
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()

    # ------------------------------------------------------------------
    # Observation toggles / task shaping
    # ------------------------------------------------------------------
    arm_dr_range = 0.3
    obs_lin_vel = True
    obs_ang_vel = True
    obs_proj_grav = True
    obs_roll = True
    obs_baseheight = False
    obs_footangle = False
    obs_notrelease = True
    obs_estimdisplace = True
    r_throw_thresh = 0.5
    r_stability_thresh = 0.22

    # ------------------------------------------------------------------
    # Target placement controls
    # ------------------------------------------------------------------
    target_fov_deg = 30.0
    target_height_range = (0.1, 1.0)
    # Rotate target heading relative to the world forward (+X).
    target_heading_offset_deg = 0.0
    robot_yaw_offset_deg = 0.0

    # ------------------------------------------------------------------
    # Misc experimental flags
    # ------------------------------------------------------------------
    distance_throw = False
    arm_only = False
    use_stability = False  # can be disabled when focusing only on throw
    no_proj_motion = False
    nonsparse_stability_reward = True
    max_throw_dist = 8


# Example of a rough-terrain variant if you want to bring back COBBLESTONE_ROAD_CFG
# @configclass
# class ThrowingG1GeneralRoughEnvCfg(ThrowingGeneralEnvCfg):
#     terrain = TerrainImporterCfg(
#         prim_path="/World/ground",
#         terrain_type="generator",
#         terrain_generator=COBBLESTONE_ROAD_CFG,
#         collision_group=-1,
#         physics_material=sim_utils.RigidBodyMaterialCfg(
#             friction_combine_mode="multiply",
#             restitution_combine_mode="multiply",
#             static_friction=1.0,
#             dynamic_friction=1.0,
#         ),
#         visual_material=sim_utils.MdlFileCfg(
#             mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
#                      f"TilesMarbleSpiderWhiteBrickBondHoned/"
#                      f"TilesMarbleSpiderWhiteBrickBondHoned.mdl",
#             project_uvw=True,
#             texture_scale=(0.1, 0.1),
#         ),
#         debug_vis=True,
#     )
