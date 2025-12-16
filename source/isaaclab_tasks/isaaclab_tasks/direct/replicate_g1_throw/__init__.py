# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym environment registration for the Replicate G1 Throw task.

This file is responsible *only* for registering the environment with Gymnasium.
No environment logic, physics, or reward definitions live here.

For KT purposes:
- Think of this as the "entry-point wiring" layer.
- Training scripts refer to the environment **by string ID**, and Gym resolves
  everything else using the information provided below.
"""

import gymnasium as gym

# Import agent-specific configuration modules (PPO configs, etc.)
from . import agents

##
# Register Gym environments.
#
# This makes the environment discoverable via:
#   gym.make("Isaac-Replicate-G1-Throw-Direct-v0")
#
# The actual environment implementation, configs, and RL-framework-specific
# settings are resolved lazily via the entry points below.
##

gym.register(
    # Unique environment ID used by Gym / IsaacLab training scripts
    id="Isaac-Replicate-G1-Throw-Direct-v0",

    # Python entry point for the environment class itself
    # This should resolve to a subclass of IsaacLab's DirectRLEnv
    entry_point=f"{__name__}.replicate_g1_throw_env:ReplicateG1ThrowEnv",

    # Disable Gym's default environment checker.
    # IsaacLab environments typically manage their own validation.
    disable_env_checker=True,

    # Additional keyword arguments passed to the environment constructor.
    # These are *paths*, not loaded objects.
    kwargs={
        # Environment configuration (scene, robot, rewards, observations, etc.)
        "env_cfg_entry_point": (
            f"{__name__}.replicate_g1_throw_env_cfg:ReplicateG1ThrowEnvCfg"
        ),

        # RL-Games PPO configuration (YAML-based)
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",

        # RSL-RL PPO configuration (Python dataclass-based)
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"
        ),

        # SKRL AMP configuration (YAML-based, adversarial motion priors)
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",

        # SKRL standard PPO configuration
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",

        # Stable-Baselines3 PPO configuration
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

# -----------------------------------------------------------------------------
# Debug / reference notes (NOT used programmatically)
# -----------------------------------------------------------------------------
# The following commented list appears to be a dump of link or joint names
# present in the G1/Alpha-style robot USD or URDF.
#
# This is useful during:
# - Contact sensor debugging
# - Collision filtering
# - Identifying hand / finger link naming conventions
#
# It is intentionally left here as a reference for developers and KT,
# but it has no effect on environment registration.
#
# ['PLINTH', 'TL1', 'LL1', 'LL2', 'LL3', 'LL4', 'LL5', 'LL6', 'LHAND',
#  'left_palm', 'left_index_proximal', 'left_index_distal',
#  'left_rib_1', 'left_rib_2', 'left_rib_3',
#  'left_pinky_proximal', 'left_pinky_distal',
#  'left_rpb_1', 'left_rpb_2', 'left_rpb_3',
#  'left_thumb_proximal', 'left_thumb_distal',
#  'left_rtb_1', 'left_rtb_2', 'left_rtb_3',
#  'left_rlb_1', 'left_rlb_2', 'left_rlb_3',
#  'left_rub_1', 'left_rub_2', 'left_rub_3', 'left_rub_4', 'left_rub_5',
#  'LHF1',
#  'RL1', 'RL2', 'RL3', 'RL4', 'RL5', 'RL6', 'RHAND',
#  'right_palm', 'right_index_proximal', 'right_index_distal',
#  'R_rpb_1', 'R_rpb_2', 'R_rpb_3',
#  'right_pinky_proximal', 'right_pinky_distal',
#  'R_rib_1', 'R_rib_2', 'R_rib_3',
#  'right_thumb_proximal', 'right_thumb_distal',
#  'R_rtb_1', 'R_rtb_2', 'R_rtb_3',
#  'R_rlb_1', 'R_rlb_2', 'R_rlb_3',
#  'R_rub_1', 'R_rub_2', 'R_rub_3', 'R_rub_4', 'R_rub_5',
#  'RHF1']
