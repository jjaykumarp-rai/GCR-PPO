# alpha_throw_env_cfg.py

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# --------------------------------------------------------------------------------------
# You should replace this with your actual Alpha ArticulationCfg
# (USD/URDF path, actuator cfg, etc.)
# --------------------------------------------------------------------------------------

ALPHA_UPPER_BODY_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Alpha",
    spawn=sim_utils.UsdFileCfg(  # or UrdfFileCfg if you're spawning URDF directly
        usd_path="/home/jjaykumarp/Projects/alpha_prims/source/alpha_ball_throw/assets/robots/alpha_sim.usd",
    ),
    # Optional: initial root state, actuators, etc. if you already have them
)

# --------------------------------------------------------------------------------------
# Events (very lightweight here – just friction/mass randomization if you want it)
# --------------------------------------------------------------------------------------

@configclass
class EventCfg:
    """Randomization & perturbations."""

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
            "asset_cfg": SceneEntityCfg("robot", body_names="PLINTH|TL1"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )


@configclass
class AlphaThrowEnvCfg(DirectRLEnvCfg):
    # -------------------------------------------------------
    # Core RL settings
    # -------------------------------------------------------
    episode_length_s = 2.0
    decimation = 4

    # 15 joints: TJ1 + RJ1–RJ7 + LJ1–LJ7
    action_scale = 0.5
    action_space = 15

    # We'll compute exact obs dim in the env at runtime,
    # but this value is used by some frameworks.
    observation_space = 100
    state_space = 0

    # Drag model flag (we keep it for future use)
    air_resistance = False

    # -------------------------------------------------------
    # Simulation
    # -------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
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
                f"{ISAACLAB_NUCLEUS_DIR}"
                "/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.1, 0.1),
        ),
        debug_vis=False,
    )

    # -------------------------------------------------------
    # Ball & target
    # -------------------------------------------------------
    sphere_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.023,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                rigid_body_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.085),  # baseball-ish
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target",
        spawn=sim_utils.CylinderCfg(
            radius=0.15,
            height=0.01,
            axis="X",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                rigid_body_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    )

    # -------------------------------------------------------
    # Scene
    # -------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,  # you can crank this up later
        env_spacing=4.0,
        replicate_physics=True,
    )

    # -------------------------------------------------------
    # Events
    # -------------------------------------------------------
    events: EventCfg = EventCfg()

    # -------------------------------------------------------
    # Sensors
    # -------------------------------------------------------
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Alpha/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # -------------------------------------------------------
    # Robot
    # -------------------------------------------------------
    robot: ArticulationCfg = ALPHA_UPPER_BODY_CFG

    # -------------------------------------------------------
    # Reward scales
    # -------------------------------------------------------
    throwing_reward_scale = 2.0
    action_rate_reward_scale = -1e-3
    joint_torque_reward_scale = -2.5e-6
    joint_accel_reward_scale = -2.5e-8
    collision_penalty_scale = 5.0  # per-step, plus termination

    # -------------------------------------------------------
    # Observations toggles
    # -------------------------------------------------------
    obs_lin_vel = True
    obs_ang_vel = True
    obs_proj_grav = True
    obs_notrelease = True
    obs_estimdisplace = True

    # -------------------------------------------------------
    # Target placement controls
    # -------------------------------------------------------
    target_fov_deg = 30.0
    target_height_range = (0.1, 1.2)
    target_heading_offset_deg = 0.0
    robot_yaw_offset_deg = 0.0

    # Horizontal distance sampling
    max_throw_dist = 8.0
