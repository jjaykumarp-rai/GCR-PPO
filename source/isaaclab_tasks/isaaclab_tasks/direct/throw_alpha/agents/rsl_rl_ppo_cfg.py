from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # ---------------- Rollouts / Training ----------------
    num_steps_per_env = 128          # was 32
    max_iterations = 12000
    save_interval = 50
    experiment_name = "throw_alpha"
    empirical_normalization = True   # keep this on

    # ---------------- Policy Network ----------------
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[256, 128],   # smaller but still expressive
        critic_hidden_dims=[256, 128],
        activation="elu",
    )

    # ---------------- PPO Hyperparameters ----------------
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,        # slightly lower entropy
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,      # a bit higher; reduce if unstable
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,           # allow somewhat larger updates
        max_grad_norm=1.0,
    )
