"""Alpha robot asset + actuator configuration for IsaacLab.

This module defines:
- Joint naming conventions and joint groups (torso, arms, fingers).
- A standard impedance-style PD actuator configuration for the torso + both arms.
- The `ArticulationCfg` used to spawn the Alpha USD into Isaac Sim / IsaacLab.

Notes for the team:
- Fingers are *not* configured in the actuator here (they are controlled elsewhere in the env).
- The USD path can be overridden via the `ALPHA_USD_PATH` environment variable.
"""

import os

from isaaclab.actuators.actuator_cfg import PDActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

# -----------------------------------------------------------------------------
# Joint limits metadata (used primarily here to populate torque limits).
#
# Values appear to be in:
# - lower/upper: radians
# - vel: rad/s
# - tau: Nm
#
# Only a subset of joints are defined here (torso + arms). Fingers are excluded.
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Joint groups for actions / control.
#
# These lists define naming conventions expected in the USD/URDF.
# They are used to:
# - construct actuator joint lists
# - keep control groupings consistent across env code
# -----------------------------------------------------------------------------

# 1) Torso DOF(s)
TORSO_JOINTS = ["TJ1"]

# 2) Right arm joints (throwing arm)
RIGHT_ARM_JOINTS = ["RJ1", "RJ2", "RJ3", "RJ4", "RJ5", "RJ6", "RJ7"]

# 3) Left arm joints (mirrored naming convention)
LEFT_ARM_JOINTS = ["LJ1", "LJ2", "LJ3", "LJ4", "LJ5", "LJ6", "LJ7"]
# If the left arm is not present / not needed, you could set this to [] in the future.

# 4) Finger joints (both hands)
# Note: finger joints are intentionally excluded from the actuator created below.
FINGER_JOINTS = [
    "RTJ1", "RIJ1", "RPJ1", "RTJ2", "RIJ2", "RPJ2",
    "LTJ1", "LIJ1", "LPJ1", "LTJ2", "LIJ2", "LPJ2",
]


def _make_alpha_impedance_actuator() -> PDActuatorCfg:
    """Create the standard impedance-style PD actuator config for Alpha.

    This actuator targets the torso + both arms using per-joint stiffness/damping
    gains and effort (torque) limits.

    Important:
    - Finger joints are NOT part of this actuator configuration.
      They are expected to be controlled separately at the environment level.

    Returns:
        PDActuatorCfg: IsaacLab actuator configuration used in `ALPHA_CFG`.
    """
    # Actuator will apply to all torso + arm joints.
    joint_names = TORSO_JOINTS + RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS

    # Per-joint proportional gains (stiffness).
    # These values define how strongly the controller pulls a joint toward its target.
    stiffness = {
        # torso
        "TJ1": 937.0,
        # right arm
        "RJ1": 150.0,
        "RJ2": 167.0,
        "RJ3": 167.0,
        "RJ4": 167.0,
        "RJ5": 13.4,
        "RJ6": 3.25,
        "RJ7": 3.25,
        # left arm (mirrored)
        "LJ1": 150.0,
        "LJ2": 167.0,
        "LJ3": 167.0,
        "LJ4": 167.0,
        "LJ5": 13.4,
        "LJ6": 3.25,
        "LJ7": 3.25,
    }

    # Per-joint derivative gains (damping).
    # These values resist motion / reduce oscillation near the target.
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

    # Effort limits (torque limits) per joint.
    # If a joint is missing from JOINT_LIMITS, it falls back to RJ5's torque limit.
    torque_limits = {
        name: JOINT_LIMITS.get(name, JOINT_LIMITS["RJ5"])["tau"]
        for name in joint_names
    }

    # Build and return the actuator config.
    return PDActuatorCfg(
        # `joint_names_expr` can take a list; IsaacLab treats these as the controlled joints.
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
        # Armature adds a small rotor inertia term for numerical stability / smoothing.
        armature=0.01,
        effort_limit=torque_limits,
    )


# -----------------------------------------------------------------------------
# USD path resolution.
#
# Priority:
# 1) Environment variable override: ALPHA_USD_PATH
# 2) Default: ~/alpha_usd/alpha/alpha.usd
# -----------------------------------------------------------------------------
_default_alpha_usd = os.path.expanduser("~/alpha_usd/alpha/alpha.usd")
ALPHA_USD = os.environ.get("ALPHA_USD_PATH", _default_alpha_usd)

# Fail fast if the USD doesn't exist; this prevents confusing downstream sim errors.
if not os.path.isfile(ALPHA_USD):
    raise FileNotFoundError(
        f"Alpha USD not found at '{ALPHA_USD}'. Set ALPHA_USD_PATH to the alpha.usd file."
    )

# -----------------------------------------------------------------------------
# Main Articulation configuration for Alpha.
#
# This is the object other modules import to spawn Alpha in an IsaacLab scene.
# -----------------------------------------------------------------------------
ALPHA_CFG = ArticulationCfg(
    # Regex-style prim path: matches one robot per environment instance.
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        # Source USD for the robot
        usd_path=ALPHA_USD,
        # Copy the USD into the stage (useful for instancing or editable overrides)
        copy_from_source=True,
        # Path under which visual materials are authored/overridden
        visual_material_path="material",
        # Enable contact sensor generation for links (required if env uses ContactSensor)
        activate_contact_sensors=True,
        # Rigid-body physics parameters applied to links
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=None,
            max_angular_velocity=None,
            max_depenetration_velocity=1.0,
        ),
        # Articulation-wide solver / collision parameters
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    # Actuators dictionary: key is a user-defined name, value is actuator config
    actuators={
        "alpha_arm": _make_alpha_impedance_actuator(),
    },
)
