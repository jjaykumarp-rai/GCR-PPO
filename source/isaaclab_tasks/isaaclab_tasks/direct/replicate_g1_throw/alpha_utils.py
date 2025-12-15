from isaaclab.actuators.actuator_cfg import PDActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

JOINT_LIMITS = {
    "TJ1": {"lower": -2.3911010752322314765, "upper": 2.3911010752322314765, "vel": 31.57, "tau": 205.483},
    "RJ1": {"lower": -0.785398163397448279, "upper": 3.0368728984701331974, "vel": 66.2, "tau": 49.0},
    "RJ2": {"lower": -3.2812189937493396741, "upper": 0.13962634015954636379, "vel": 66.2, "tau": 49.0},
    "RJ3": {"lower": -2.5132741228718344928, "upper": 0.92502450355699461504, "vel": 66.2, "tau": 49.0},
    "RJ4": {"lower": 0.0, "upper": 0.5830872929516077718, "vel": 66.2, "tau": 49.0},
    "RJ5": {"lower": -1.5184364492350665987, "upper": 1.5184364492350665987, "vel": 89.0, "tau": 35.0},
    "RJ6": {"lower": -1.2217304763960306069, "upper": 1.2217304763960306069, "vel": 200.0, "tau": 12.0},
    "RJ7": {"lower": -0.349066, "upper": 0.349066, "vel": 200.0, "tau": 12.0},
    "LJ1": {"lower": -0.785398163397448279, "upper": 3.0368728984701331974, "vel": 66.2, "tau": 49.0},
    "LJ2": {"lower": -3.2812189937493396741, "upper": 0.13962634015954636379, "vel": 66.2, "tau": 49.0},
    "LJ3": {"lower": -2.5132741228718344928, "upper": 0.92502450355699461504, "vel": 66.2, "tau": 49.0},
    "LJ4": {"lower": 0.0, "upper": 0.5830872929516077718, "vel": 66.2, "tau": 49.0},
    "LJ5": {"lower": -1.5184364492350665987, "upper": 1.5184364492350665987, "vel": 89.0, "tau": 35.0},
    "LJ6": {"lower": -1.2217304763960306069, "upper": 1.2217304763960306069, "vel": 200.0, "tau": 12.0},
    "LJ7": {"lower": -0.349066, "upper": 0.349066, "vel": 200.0, "tau": 12.0},
}

# -----------------------------
# Joint groups for actions
# -----------------------------

# 1) Torso DOF(s)
TORSO_JOINTS = ["TJ1"]

# 2) Right arm joints (throwing arm)
RIGHT_ARM_JOINTS = ["RJ1", "RJ2", "RJ3", "RJ4", "RJ5", "RJ6", "RJ7"]

# 3) Left arm joints
# If your left arm joints are named LJ1..LJ7 in the URDF/USD, use this:
LEFT_ARM_JOINTS = ["LJ1", "LJ2", "LJ3", "LJ4", "LJ5", "LJ6", "LJ7"]
# If they don't exist yet / you don't care for now, you can temporarily do:
# LEFT_ARM_JOINTS = []

# 4) Finger joints (BOTH hands, from your greps)
#   RTJ* / RIJ* / RPJ* / LPJ*
FINGER_JOINTS = [
    "RTJ1", "RIJ1", "RPJ1", "RTJ2", "RIJ2", "RPJ2",
    "LTJ1", "LIJ1", "LPJ1", "LTJ2", "LIJ2", "LPJ2",
]


def _make_alpha_impedance_actuator() -> PDActuatorCfg:
    """Standard PD actuator for torso + right arm + left arm.
    Fingers are controlled separately in the env (not here).
    """
    joint_names = TORSO_JOINTS + RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS

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
        "TJ1": 4.7,
        "RJ1": 3.1,
        "RJ2": 6.6,
        "RJ3": 6.6,
        "RJ4": 6.6,
        "RJ5": 0.3,
        "RJ6": 0.3,
        "RJ7": 0.3,
        "LJ1": 3.1,
        "LJ2": 6.6,
        "LJ3": 6.6,
        "LJ4": 6.6,
        "LJ5": 0.3,
        "LJ6": 0.3,
        "LJ7": 0.3,
    }

    torque_limits = {
        name: JOINT_LIMITS.get(name, JOINT_LIMITS["RJ5"])["tau"]
        for name in joint_names
    }

    return PDActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
        armature=0.01,
        effort_limit=torque_limits,
    )


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
