# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Replicate-G1-Throw-Direct-v0",
    entry_point=f"{__name__}.replicate_g1_throw_env:ReplicateG1ThrowEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.replicate_g1_throw_env_cfg:ReplicateG1ThrowEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

# ['PLINTH', 'TL1', 'LL1', 'LL2', 'LL3', 'LL4', 'LL5', 'LL6', 'LHAND', 'left_palm', 'left_index_proximal', 'left_index_distal', 'left_rib_1', 'left_rib_2', 'left_rib_3', 'left_pinky_proximal', 'left_pinky_distal', 'left_rpb_1', 'left_rpb_2', 'left_rpb_3', 'left_thumb_proximal', 'left_thumb_distal', 'left_rtb_1', 'left_rtb_2', 'left_rtb_3', 'left_rlb_1', 'left_rlb_2', 'left_rlb_3', 'left_rub_1', 'left_rub_2', 'left_rub_3', 'left_rub_4', 'left_rub_5', 'LHF1', 'RL1', 'RL2', 'RL3', 'RL4', 'RL5', 'RL6', 'RHAND', 'right_palm', 'right_index_proximal', 'right_index_distal', 'R_rpb_1', 'R_rpb_2', 'R_rpb_3', 'right_pinky_proximal', 'right_pinky_distal', 'R_rib_1', 'R_rib_2', 'R_rib_3', 'right_thumb_proximal', 'right_thumb_distal', 'R_rtb_1', 'R_rtb_2', 'R_rtb_3', 'R_rlb_1', 'R_rlb_2', 'R_rlb_3', 'R_rub_1', 'R_rub_2', 'R_rub_3', 'R_rub_4', 'R_rub_5', 'RHF1']