# throw_alpha_env_cfg.py
#
# Self-contained Alpha throwing env config:
# - No G1 imports
# - Defines ALPHA_CFG and ThrowingAlphaGeneralEnvCfg in this file

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen

from isaaclab.actuators.actuator_cfg import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# --------------------------------------------------------------------------
# Alpha joint limits + impedance profile
# --------------------------------------------------------------------------

JOINT_LIMITS = {
    "TJ1": {"lower": -2.3911010752322314765, "upper": 2.3911010752322314765, "vel": 31.57, "tau": 205.483},
    "RJ1": {"lower": -0.785398163397448279, "upper": 3.0368728984701331974, "vel": 66.2, "tau": 49.0},
    "RJ2": {"lower": -3.2812189937493396741, "upper": 0.13962634015954636379, "vel": 66.2, "tau": 49.0},
    "RJ3": {"lower": -2.5132741228718344928, "upper": 0.92502450355699461504, "vel": 66.2, "tau": 49.0},
    "RJ4": {"lower": 0.0, "upper": 0.5830872929516077718, "vel": 66.2, "tau": 49.0},
    "RJ5": {"lower": -1.5184364492350665987, "upper": 1.5184364492350665987, "vel": 89.0, "tau": 35.0},
    "RJ6": {"lower": -1.2217304763960306069, "upper": 1.2217304763960306069, "vel": 200.0, "tau": 12.0},
    "RJ7": {"lower": -0.349066, "upper": 0.349066, "vel": 200.0, "tau": 12.0},
}

# -----------------------------
# Joint groups for actions
# -----------------------------

# 1) Torso DOF(s)
TORSO_JOINTS = ["TJ1"]

# 2) Right arm joints
RIGHT_ARM_JOINTS = ["RJ1", "RJ2", "RJ3", "RJ4", "RJ5", "RJ6", "RJ7"]

# 3) Left arm joints
# ⚠️ IMPORTANT: update these names to match your actual USD joint names.
LEFT_ARM_JOINTS = ["LJ1", "LJ2", "LJ3", "LJ4", "LJ5", "LJ6", "LJ7"]


def _make_alpha_impedance_actuator() -> DelayedPDActuatorCfg:
    """Impedance (Delayed PD) actuator for torso + right arm + left arm."""

    # Full list of controlled joints (in any order)
    joint_names = TORSO_JOINTS + RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS

    # Reuse right-arm gains for left arm (you can tune separately later)
    stiffness = {
        # torso
        "TJ1": 937.0,
        # right arm
        "RJ1": 295.0,
        "RJ2": 334.0,
        "RJ3": 334.0,
        "RJ4": 334.0,
        "RJ5": 26.8,
        "RJ6": 6.5,
        "RJ7": 6.5,
        # left arm (mirrored)
        "LJ1": 295.0,
        "LJ2": 334.0,
        "LJ3": 334.0,
        "LJ4": 334.0,
        "LJ5": 26.8,
        "LJ6": 6.5,
        "LJ7": 6.5,
    }

    damping = {
        # torso
        "TJ1": 4.7,
        # right arm
        "RJ1": 3.1,
        "RJ2": 6.6,
        "RJ3": 6.6,
        "RJ4": 6.6,
        "RJ5": 0.3,
        "RJ6": 0.3,
        "RJ7": 0.3,
        # left arm (mirrored)
        "LJ1": 3.1,
        "LJ2": 6.6,
        "LJ3": 6.6,
        "LJ4": 6.6,
        "LJ5": 0.3,
        "LJ6": 0.3,
        "LJ7": 0.3,
    }

    # If a joint is not in JOINT_LIMITS (e.g., left arm), fall back to RJ5 tau
    torque_limits = {
        name: JOINT_LIMITS.get(name, JOINT_LIMITS["RJ5"])["tau"]
        for name in joint_names
    }

    return DelayedPDActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
        armature=0.01,
        effort_limit=torque_limits,
        min_delay=1,
        max_delay=1,
    )


# --------------------------------------------------------------------------
# Alpha articulation config
# --------------------------------------------------------------------------

ALPHA_USD = "/home/jjaykumarp/Projects/my_alpha_usd/alpha/alpha.usd"

ALPHA_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ALPHA_USD,
        copy_from_source=True,
        visual_material_path="material",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=None,
            max_angular_velocity=None,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    actuators={
        "alpha_arm": _make_alpha_impedance_actuator(),
    },
)

# --------------------------------------------------------------------------
# Terrain
# --------------------------------------------------------------------------

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
    """Configuration for randomization."""

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

    # ⚠️ Alpha has no 'imu_link' – use PLINTH for base mass randomization
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="PLINTH"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )


@configclass
class ThrowingAlphaGeneralEnvCfg(DirectRLEnvCfg):
    """Alpha ball throwing config (ported from ThrowingGeneralEnvCfg)."""

    # env
    episode_length_s = 2.0
    decimation = 4
    action_scale = 0.5
    # Actions: [ torso (1), right arm (7), left arm (7), grip (1) ] = 16
    action_space = 16
    observation_space = 100
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
            radius=0.15,
            height=0.01,
            axis="X",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                rigid_body_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
            ),
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # events
    events: EventCfg = EventCfg()

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # Use Alpha robot instead of G1
    robot: ArticulationCfg = ALPHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")


    # ------------------------------------------------------------------
    # Rewards: base scales
    # ------------------------------------------------------------------
    throwing_reward_scale: float = 3.0
    roll_reward_scale: float = 0.5
    stability_reward_scale: float = 0.25
    throw_height_reward_scale: float = 2.0
    action_rate_reward_scale: float = -1e-3
    joint_torque_reward_scale: float = -2.5e-6
    joint_accel_reward_scale: float = -2.5e-8

    # optional reward term scales
    baseh_reward_scale: float = 0.5
    energy_reward_scale: float = 2.0
    ballrel_reward_scale: float = 1.5
    bodymo_reward_scale: float = 1.0
    lftarm_reward_scale: float = 1.0
    rgtarmrel_reward_scale: float = 2.0

    use_roll_reward: bool = True
    use_stability_reward: bool = True

    # ------------------------------------------------------------------
    # Observations / thresholds
    # ------------------------------------------------------------------
    arm_dr_range: float = 0.3
    obs_lin_vel: bool = True
    obs_ang_vel: bool = True
    obs_proj_grav: bool = True
    obs_roll: bool = True
    obs_baseheight: bool = False
    obs_footangle: bool = False
    obs_notrelease: bool = True
    obs_estimdisplace: bool = True

    r_throw_thresh: float = 0.5
    r_stability_thresh: float = 0.22

    # ------------------------------------------------------------------
    # Target placement
    # ------------------------------------------------------------------
    target_fov_deg: float = 30.0
    target_height_range: tuple[float, float] = (0.1, 1.0)
    target_heading_offset_deg: float = 0.0
    robot_yaw_offset_deg: float = 0.0

    # ------------------------------------------------------------------
    # Mode flags
    # ------------------------------------------------------------------
    distance_throw: bool = False
    arm_only: bool = False
    use_stability: bool = False
    no_proj_motion: bool = False
    nonsparse_stability_reward: bool = True
    max_throw_dist: int = 8

    # toggles for extra reward terms
    baseh_rew: bool = False
    energy_rew: bool = True
    ballrel_rew: bool = True
    bodymo_rew: bool = True
    lftarm_rew: bool = True
    rgtarmrel_rew: bool = True