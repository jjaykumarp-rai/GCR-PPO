# Replicate G1 Throw Task Reference

This document summarizes the key observation/action/reward/curriculum details for the `Isaac-Replicate-G1-Throw-Direct-v0` task along with the control/physics cadence.

## Observations

The observation vector (`obs_dim = 105`) is assembled in `_get_observations` and contains the following components (each gated by Hydra flags):

- Base states: optional angular velocity (`obs_ang_vel`), linear velocity (`obs_lin_vel`), projected gravity (`obs_proj_grav`), and optional roll angle (`obs_roll`).
- Throwing command tuple `(dist, theta_cos, phi)` that encodes the sampled target location.
- Joint proprioception: non-finger joint position offsets from their default pose plus optional noise (`joint_pos_noise_range`), and joint velocities plus noise (`joint_vel_noise_range`).
- Previous control actions (16-dimensional action vector) and the `not_released_ball` flag (`obs_notrelease`) so the policy is aware of the release phase.
- Projectile estimates (`obs_estimdisplace`) including normalized displacement to the target, remaining intercept time, and raw ball velocity when enabled (`obs_estimdisplace` also populates an estimate of the ball’s velocity components).

Noise and curriculum shaping gradually inject additional randomness into these estimates via `action_noise`.

## Actions

- 16-dimensional action space: torso (1 DoF), right arm (7 DoF), left arm (7 DoF), and a grip scalar that switches between closed (≤0) and open (>0).
- Actions represent joint position deltas around a stored desired pose and are scaled by `action_scale = 0.5`.
- Control decimation is set to `decimation = 4`, so the policy outputs commands at 50 Hz while the simulation runs at 200 Hz (`sim.dt = 1/200`).
- Finger commands are interpreted as discrete open/close presets rather than continuous positions.

## Rewards

Reward components are collected into `self.reward_component_names` and logged via `_episode_sums`. Default scales defined in `replicate_g1_throw_env_cfg.py` include:

| Component | Description | Default Scale |
|-----------|-------------|---------------|
| `throwing` | Target accuracy reward once the ball hits or overflies the board. | 3.0 |
| `projectile_rew` | Projectile shaping that encourages aiming the flight path toward the sampled command. | 0.5 (`projectile_reward_scale`) |
| `stability` | Sparse stability reward that penalizes collisions and premature drops. | 0.25 |
| `action_rate_l2` | Penalty on action rate change (smoothness). | −1e-3 |
| `dof_torques_l2` | Torque penalty from applied joint torques. | −2.5e-6 |
| `dof_acc_l2` | Joint acceleration penalty. | −2.5e-8 |
| `ballrel_rew` | One-shot reward when the ball crosses the release threshold. | 0.5 (`ball_release_reward_scale`) |
| `fingers_not_blocking` | Reward that encourages the throw direction to move away from the grip by rewarding a non-zero X-component of the throw vector in the end-effector frame. | 0.5 (`fingers_not_blocking_reward_scale`, with `fingers_not_blocking_sigma = 0.1`) |
| `hand_recontact` | Penalty applied if the hand re-contacts the ball after release (also forces episode termination). | −1.0 (`hand_recontact_penalty_scale`) |

Additional penalties include `action_limit_penalty_scale = −1e-3`, `joint_vel_reward_scale = −1e-4`, and `joint_accel_reward_scale = −2.5e-8` that act on their corresponding signals.

## Curriculum

- Distance sampling starts at `min_throw_dist = 1` and initially caps at `initial_max_throw_dist` (if provided) or `max_throw_dist = 8`.
- The upper distance bound expands over time during `update_curriculum` once both throwing and stability rewards cross thresholds (`r_throw_thresh = 0.5`, `r_stability_thresh = 0.22`) or after 750 iterations. The increment size is tuned via `curriculum_distance_increment` (default 0.01), and the height window grows by `curriculum_height_increment`.
- Target height range is bounded by `target_height_range = (0.1, 1.0)` and only expands upward as the curriculum progresses (`current_target_height_range`).
- `theta_range` is extended toward the full [-1, 1] range (governed by `self.theta_range`) as the curriculum unlocks wider polar angles.
- `action_noise` drifts from 0 toward 1 to gradually introduce randomness into the projectile displacement estimates used by the policy.

## Termination

- Episodes terminate (`terminated = True`) when:
  - The ball hits the ground (`ball_pos.z < 0.05`) after release.
  - The ball strays too far from the target board (`distance_xy > max_throw_dist + 1.0`).
  - The agent never releases before a timeout fraction of the max episode length (`no_release_timeout_frac = 0.75`).
  - A joint velocity limit is exceeded (if `joint_velocity_limit` is set) or the hand recontacts the ball post-release.
- Episodes truncate (`truncated = True`) once the configured episode length is reached (`episode_length_s = 2.0`, `max_episode_length = episode_length_s * control_rate`).
- Success is logged when a released ball both hits the visual target disk and contacts the ground within `success_radius = 0.55` meters.
- Hit counters track board hits and close ground impacts separately for logging/debug (`_target_hit_label`, `_ground_hit_label`).

## Physics & Model Frequencies

- Simulation runs at 200 Hz (`SimulationCfg.dt = 1/200`) with the robot physics, contact sensors, and ball dynamics stepping at this rate.
- Control / policy actions are executed every 4 simulation steps (`decimation = 4`), resulting in a 50 Hz control/model frequency that the policy observes and acts upon.

