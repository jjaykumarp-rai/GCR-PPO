# rsl_rl_ppo_cfg.py
#
# PPO config for the Alpha throwing task (ThrowingAlphaEnv).
# Tweak num_steps_per_env, max_iterations, network size, and lr as needed.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # ------------------------------------------------------------------
    # Rollout / training schedule
    # ------------------------------------------------------------------
    # How many sim steps per env before one PPO update
    num_steps_per_env = 32

    # Total PPO iterations (so total env steps = num_envs * num_steps_per_env * max_iterations)
    max_iterations = 12000

    # Save policy every N iterations
    save_interval = 50

    # This name shows up in the log directory:
    # logs/rsl_rl/throw_alpha/<timestamp>/...
    experiment_name = "throw_alpha"

    # Use running obs/std normalization (helps with large obs vectors)
    empirical_normalization = True

    # ------------------------------------------------------------------
    # Policy network (Actor / Critic)
    # ------------------------------------------------------------------
    policy = RslRlPpoActorCriticCfg(
        # Initial std dev for Gaussian policy
        init_noise_std=0.8,
        # Throwing obs is fairly high-dim; use a bigger MLP than cartpole
        actor_hidden_dims=[512, 256],
        critic_hidden_dims=[512, 256],
        activation="elu",
    )

    # ------------------------------------------------------------------
    # PPO algorithm hyperparameters
    # ------------------------------------------------------------------
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,          # a bit more entropy to encourage exploration
        num_learning_epochs=5,
        num_mini_batches=4,         # can bump to 8 if you have many envs
        learning_rate=3.0e-4,       # smaller than cartpole's 1e-3
        schedule="adaptive",        # KL-based adaptive LR
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
