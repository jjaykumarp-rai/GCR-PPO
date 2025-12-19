# 🦾🎯 Replicate G1 Throw Task Reference

This document summarizes the key **observation, action, reward, and curriculum**
details for the `Isaac-Replicate-G1-Throw-Direct-v0` task, along with the
**control and physics cadence**.

---

## 👀 Observations

The observation vector (**`obs_dim = 105`**) is assembled in `_get_observations`
and contains the following components (each gated by Hydra flags):

### 🧭 Base States
- Optional angular velocity (`obs_ang_vel`)
- Linear velocity (`obs_lin_vel`)
- Projected gravity (`obs_proj_grav`)
- Optional roll angle (`obs_roll`)

### 🎯 Throwing Command
- Tuple **`(dist, theta_cos, phi)`** encoding the sampled target location

### 🦿 Joint Proprioception
- Non-finger joint position offsets from default pose  
  (+ optional noise: `joint_pos_noise_range`)
- Joint velocities  
  (+ optional noise: `joint_vel_noise_range`)

### 🔁 Previous Actions
- Previous control action (16-D)
- `not_released_ball` flag (`obs_notrelease`) indicating release phase

### 🏀 Projectile Estimates (`obs_estimdisplace`)
- Normalized displacement to target
- Remaining intercept time
- Raw ball velocity (when enabled)
- Estimated ball velocity components

🎲 Noise and curriculum shaping progressively inject randomness into these
estimates via `action_noise`.

---

## 🎮 Actions

- **16-dimensional action space**
  - Torso: 1 DoF
  - Right arm: 7 DoF
  - Left arm: 7 DoF
  - Grip scalar (≤ 0 = closed, > 0 = open)

- Actions represent joint position deltas around a stored desired pose  
  and are scaled by `action_scale = 0.5`

- **⏱ Control decimation**
  - `decimation = 4`
  - Policy frequency: **50 Hz**
  - Simulation frequency: **200 Hz** (`sim.dt = 1/200`)

- **✋ Finger control**
  - Discrete open / close presets
  - Not continuous joint positions

---

## 🏆 Rewards

Reward components are collected into `self.reward_component_names`
and logged via `_episode_sums`.

### 📊 Default Reward Scales

| Component              | Description                                                                 | Default Scale |
|------------------------|-----------------------------------------------------------------------------|---------------|
| `throwing`             | Target accuracy after hit or overflight                                     | **3.0**       |
| `projectile_rew`       | Projectile shaping toward sampled command                                   | **0.5**       |
| `stability`            | Penalizes collisions and premature drops                                    | **0.25**      |
| `action_rate_l2`       | Action rate smoothness penalty                                               | −1e-3         |
| `dof_torques_l2`       | Joint torque penalty                                                        | −2.5e-6       |
| `dof_acc_l2`           | Joint acceleration penalty                                                  | −2.5e-8       |
| `ballrel_rew`          | One-shot reward when the ball is released                                   | **0.5**       |
| `fingers_not_blocking` | Encourages throw direction away from grip                                   | **0.5**       |
| `hand_recontact`       | Penalty if the hand re-contacts the ball after release (terminates episode) | **−1.0**      |

**Additional penalties**
- `action_limit_penalty_scale = −1e-3`
- `joint_vel_reward_scale = −1e-4`
- `joint_accel_reward_scale = −2.5e-8`

---

## 📈 Curriculum

### 📏 Throw Distance
- Starts at `min_throw_dist = 1`
- Initially capped at `initial_max_throw_dist`
  or `max_throw_dist = 8`

### 🚀 Progressive Expansion
- Upper bound expands when:
  - `throwing > 0.5` **and**
  - `stability > 0.22`
- Or automatically after **750 iterations**
- Increment size: `curriculum_distance_increment = 0.01`

### 🪜 Target Height
- Initial range: `(0.1, 1.0)`
- Expands upward via `curriculum_height_increment`
- Tracked by `current_target_height_range`

### 🧭 Angular Coverage
- `theta_range` gradually unlocks toward full `[-1, 1]`

### 🎲 Action Noise
- `action_noise` drifts from `0 → 1`
- Adds randomness to projectile displacement estimates

---

## ⛔ Termination

### Episodes terminate when:
- 🏀 Ball hits ground after release (`ball_pos.z < 0.05`)
- 📍 Ball strays too far from target in XY  
  (`distance_xy > max_throw_dist + 1.0`)
- ⏳ No release before timeout  
  (`no_release_timeout_frac = 0.75`)
- ⚠️ Joint velocity limit exceeded (if enabled)
- ✋ Hand re-contacts ball after release

### Episodes truncate when:
- Maximum episode length is reached  
  (`episode_length_s = 2.0`,  
  `max_episode_length = episode_length_s * control_rate`)

### ✅ Success Criteria
- Ball is released
- Ball hits the visual target disk
- Ball contacts the ground within `success_radius = 0.55 m`

### 📊 Logging Helpers
- `_target_hit_label` → board hits
- `_ground_hit_label` → close ground impacts

---

## ⚙️ Physics & Model Frequencies

### 🧪 Simulation
- Runs at **200 Hz**
- `SimulationCfg.dt = 1/200`
- Includes robot physics, contact sensors, and ball dynamics

### 🧠 Control / Policy
- Executes every **4 simulation steps**
- Effective control frequency: **50 Hz**
