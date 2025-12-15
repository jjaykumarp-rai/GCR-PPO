# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


###### part of CDR-1630 ######
# terrain specific imports
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg

# event specific imports
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

# env_cfg specific imports
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from .alpha_utils import ALPHA_CFG


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
            proportion=0.25, slope_range=(0.05,0.15), platform_width=0.0
        ),
        "slopey_inverted": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.05,0.15), platform_width=0.0, inverted=True,
        ),
    },
)


@configclass
class EventCfg:
    """Configuration for randomization."""

    # physics material randomization
    # 
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


    # arm_joints = [
    #     # Left arm joints
    #     "left_shoulder_pitch_joint",
    #     "left_shoulder_roll_joint",
    #     "left_shoulder_yaw_joint",
    #     "left_elbow_joint",
    #     "left_wrist_roll_joint",

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
    # This range is the sweet spot for throwing / fast arm motions.


    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="PLINTH"), # base_link for chuck, base for spot, imu_link for unitree
    #         "mass_distribution_params": (-0.5, 0.5),
    #         "operation": "add",
    #     },
    # )
    '''push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.0, 4.0),
        params={"velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}},
    )'''


@configclass
class ReplicateG1ThrowEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 2.0
    decimation = 4
    action_scale = 0.5
    # Actions: [ torso (1), right arm (7), left arm (7), grip (1) ] = 16
    action_space = 16
    observation_space = 105
    state_space = 0
    air_resistance = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    '''terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        #max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
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
    )'''

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
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.1, 0.1),
            ),
        debug_vis=False,
    )

    sphere_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/sphere",
        spawn= sim_utils.SphereCfg(
            radius=0.023,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False,
                disable_gravity=False,
                rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.085),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg()
    )

    target_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target",
        spawn=sim_utils.CylinderCfg(
            radius=0.15,          # disk radius (m)
            height=0.01,          # very thin -> disk-like
            axis="X",             # axis along X => faces normal to X
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                rigid_body_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),   # bright red
            ),
        ),
        # init_state=RigidObjectCfg.InitialStateCfg(
        #     pos=(4.0, 0.0, 1.0),      # in front of robot, at some height
        #     rot=(1.0, 0.0, 0.0, 0.0), # identity, since axis="X" already
        # ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=8.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # robot
    robot: ArticulationCfg = ALPHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    throwing_reward_scale = 3.0  # prioritize target accuracy
    throw_height_reward_scale = 1.0  # secondary to distance/accuracy
    throw_height_target = 0.6
    action_rate_reward_scale = -1e-3
    joint_torque_reward_scale = -2.5e-6
    joint_vel_reward_scale = -1e-4
    joint_accel_reward_scale = -2.5e-8
    joint_vel_penalty_clip = 1.0e4
    joint_accel_penalty_clip = 1.0e4
    action_limit_penalty_scale = -1e-3
    #throw_time_reward_scale = 1.0#1.0
    #zvel_reward_scale = 0.75

    arm_dr_range = 0.3
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
    r_throw_thresh = 0.5

    # target placement controls
    min_throw_dist = 3.0  # start curriculum close, then expand outward
    target_fov_deg = 30.0
    target_height_range = (0.1, 1.0)
    # Rotate target heading relative to the world forward (+X). Alpha faces +Y, so add 90 deg.
    target_heading_offset_deg = 90.0
    robot_yaw_offset_deg = 0.0

    # just for experiments...
    use_stability = True # dont need
    no_proj_motion = False
    nonsparse_stability_reward = False # so that stability reward contains only collision and ball not thrown condition
    max_throw_dist = 5

    # ---------------------------------------------------------------------
    # things I changed for migration from throwing to replicate_g1_throw
    obs_roll = False

    distance_throw = False
    arm_only = False

    # need to check if this is required
    roll_reward_scale =  0.43
    stability_reward_scale = 0.25#0.00001#1 (0.2 before)
    r_stability_thresh = 0.22 
    # ---------------------------------------------------------------------
