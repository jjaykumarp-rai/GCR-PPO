from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class ThrowingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 26
    max_iterations = 12000
    save_interval = 10
    experiment_name = "throwing"
    empirical_normalization = False

    # ⬇️ add these video-related fields
    # record every N iterations (set to 1 to record each iteration)
    save_video_interval = 50        # or 10, or 1 if you’re brave
    # how many sim steps between rendered frames in the video
    video_skip = 2                  # 1 = every step, 2 = every 2nd step
    # optionally, limit number of episodes per video
    max_video_length = 1000         # in env steps (if your version has this)

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=[768, 512, 256],
        critic_hidden_dims=[768, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.0003552,
        num_learning_epochs=5,
        num_mini_batches=6,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.92,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
