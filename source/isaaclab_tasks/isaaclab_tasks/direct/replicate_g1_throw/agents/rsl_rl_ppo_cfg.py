# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
PPO configuration for the Replicate G1 Throwing task using RSL-RL.

This file defines:
- The PPO runner configuration (training loop, rollout length, logging, etc.)
- The actor-critic network architecture
- PPO algorithm hyperparameters

This config is consumed by Isaac Lab's RSL-RL integration and passed
to the training runner when launching experiments.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    PPO Runner configuration for the Replicate G1 Throw environment.

    This class controls the *training loop level* behavior:
    - how many steps are collected per environment
    - total training iterations
    - checkpointing frequency
    - experiment naming and logging
    - PPO policy and algorithm sub-configs

    It is referenced during environment registration as:
        rsl_rl_cfg_entry_point
    """

    # ------------------------------------------------------------------
    # Rollout / training loop parameters
    # ------------------------------------------------------------------

    # Number of environment steps collected per env before each PPO update
    num_steps_per_env = 25

    # Total number of PPO iterations (not environment steps)
    max_iterations = 12000

    # Save model checkpoint every N iterations
    save_interval = 50

    # Experiment name used for logging directories
    experiment_name = "replicate_g1_throw"

    # Whether to apply empirical observation normalization
    # (disabled here, as observations are already reasonably scaled)
    empirical_normalization = False

    # Maximum video length (in env steps) when recording rollouts
    max_video_length = 2000

    # ------------------------------------------------------------------
    # Policy (Actor-Critic) network configuration
    # ------------------------------------------------------------------

    policy = RslRlPpoActorCriticCfg(
        # Initial standard deviation for action distribution
        # Higher value → more exploration early in training
        init_noise_std=0.25,

        # Hidden layer dimensions for actor network
        # (maps observations → action mean)
        actor_hidden_dims=[256, 128, 72],

        # Hidden layer dimensions for critic network
        # (maps observations → value estimate)
        critic_hidden_dims=[256, 128, 72],

        # Activation function used in both actor and critic
        activation="elu",
    )

    # ------------------------------------------------------------------
    # PPO algorithm hyperparameters
    # ------------------------------------------------------------------

    algorithm = RslRlPpoAlgorithmCfg(
        # Weight for value function loss in total PPO loss
        value_loss_coef=1.0,

        # Whether to clip value function updates (recommended for PPO)
        use_clipped_value_loss=True,

        # PPO clipping parameter (ε in clipped surrogate objective)
        clip_param=0.2,

        # Entropy bonus coefficient (encourages exploration)
        entropy_coef=0.0003552,

        # Number of optimization epochs per PPO update
        num_learning_epochs=5,

        # Number of mini-batches per epoch
        num_mini_batches=6,

        # Adam learning rate
        learning_rate=5.0e-4,

        # Learning rate schedule:
        # "adaptive" adjusts LR based on KL divergence
        schedule="adaptive",

        # Discount factor for future rewards
        gamma=0.99,

        # GAE (Generalized Advantage Estimation) lambda
        lam=0.95,

        # Target KL divergence for adaptive LR schedule
        desired_kl=0.01,

        # Gradient clipping norm for stability
        max_grad_norm=1.0,
    )
