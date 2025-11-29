import os
import torch


class Schedule:
    """
    Generic curriculum scheduler for throwing.

    - param_names: list of attribute names to modify on `parent_obj`
    - param_ranges: list of curriculum levels; each level is a list of ranges,
      one per param name.
        Example shape:
          param_ranges[level][param_idx] = [low, high]  or  [value]
        So overall: len(param_ranges) == cl_levels,
                    len(param_ranges[level]) == len(param_names)
    - converg_crit: function(list_of_reward_lists) -> bool
      Takes e.g. [past_avg_rewards_0, past_avg_rewards_1, ...] and returns
      True if we should move to the next curriculum level.
    - cl_levels: number of curriculum stages
    - num_envs: number of parallel envs (for sampling per-env values)
    - device: torch device
    - random_sample: if True, re-sample ranged params every update call,
      not just when we advance a curriculum level.
    """

    def __init__(
        self,
        param_names,
        param_ranges,
        converg_crit,
        cl_levels,
        num_envs,
        device,
        random_sample: bool = True,
    ):
        assert len(param_ranges) == cl_levels, (
            f"param_ranges must have length cl_levels; "
            f"got len(param_ranges)={len(param_ranges)}, cl_levels={cl_levels}"
        )
        assert len(param_ranges[0]) == len(param_names), (
            f"Each level in param_ranges must have len(param_names); "
            f"got len(param_ranges[0])={len(param_ranges[0])}, "
            f"len(param_names)={len(param_names)}"
        )

        self.param_names = param_names
        self.param_ranges = param_ranges
        self.converg_crit = converg_crit
        self.cl_levels = cl_levels
        self.current_level = 0
        self.last_update_iter = 0
        self.num_envs = num_envs
        self.device = device
        self.random_sample = random_sample

    def set_parameters(self, parent_obj, use_print: bool = True):
        """
        Set curriculum-controlled attributes on parent_obj.

        - If a range is a single value [v], set that scalar directly.
        - If it is [low, high], sample per-env values in that range and assign.
        """
        # For shaped params (per env), we allocate here but only use when needed.
        self.values = torch.zeros(len(self.param_ranges[0]), device=self.device)

        for i, name in enumerate(self.param_names):
            range_i = self.param_ranges[self.current_level][i]

            # Single fixed value: set scalar (or tensor) directly.
            if len(range_i) == 1:
                val = range_i[0]
                setattr(parent_obj, name, val)
                if use_print:
                    print(f"[Curriculum] Set {name} to {val} at level {self.current_level}.")

            # Range [low, high]: sample per-env values.
            else:
                low, high = range_i[0], range_i[1]
                value = torch.rand(self.num_envs, 1, device=self.device) * (high - low) + low
                setattr(parent_obj, name, value)
                if use_print:
                    print(
                        f"[Curriculum] Set {name} to range [{low}, {high}] "
                        f"(per-env sampled) at level {self.current_level}."
                    )

    def update(self, parent_obj, past_rewards, past_avg_rewards):
        """
        Update curriculum:

        - Optionally increase observation noise (for G1 throwing env) once
          enough iterations have passed and performance is still low.
        - If converg_crit is satisfied, advance to the next curriculum level
          and call set_parameters().
        - If random_sample is True, re-sample ranged parameters each call.
        """
        # ---- Noise schedule (adapted for G1 env)
        # past_avg_rewards[0] is assumed to be some main task reward curve.
        # We require at least 4000 samples before starting noise curriculum.
        if len(past_avg_rewards[0]) >= 4000:
            # We don't check cfg.env.task_throw anymore; assume this is the
            # general throwing task.
            # In our env we used `action_noise` instead of `noise_obs`.
            if past_avg_rewards[0][-1] <= 1.0 and getattr(parent_obj, "action_noise", 0.0) < 1.0:
                if self.last_update_iter < len(past_avg_rewards[0]):
                    new_noise = min(1.0, float(getattr(parent_obj, "action_noise", 0.0) + 1.0 / 100.0))
                    setattr(parent_obj, "action_noise", new_noise)
                    print(f"[Curriculum] Increased action_noise to {new_noise}.")
                    self.last_update_iter = len(past_avg_rewards[0])

        # ---- Check for curriculum level convergence
        if (
            len(past_avg_rewards[0]) > 0
            and self.converg_crit([x[self.last_update_iter:] for x in past_avg_rewards])
            and self.current_level < self.cl_levels - 1
        ):
            # Advance to next level
            self.current_level += 1
            print(f"[Curriculum] Advancing to level {self.current_level}.")
            self.set_parameters(parent_obj)
            self.last_update_iter = len(past_avg_rewards[0])

        # ---- Optionally re-sample within current level
        elif self.random_sample:
            # Re-sample parameters that are specified as ranges
            self.set_parameters(parent_obj, use_print=False)


class NoCurriculum:
    """
    Simple non-CL sampler: every update, resample parameters uniformly
    within the given ranges and set them on parent_obj.

    - param_names: list of attribute names
    - param_ranges: list with shape [num_params][2] for [low, high]
    """

    def __init__(self, param_names, param_ranges, num_envs, device):
        # Clean up optional log from older runs
        if os.path.exists("random_log.txt"):
            os.remove("random_log.txt")

        self.param_names = param_names
        self.param_ranges = param_ranges
        self.device = device
        self.num_envs = num_envs

        # Initial random draw
        self.throwing_values = torch.rand(self.num_envs, len(param_names), device=self.device)
        for i in range(len(param_names)):
            low, high = param_ranges[i][0], param_ranges[i][1]
            self.throwing_values[:, i] = self.throwing_values[:, i] * (high - low) + low

        self.last_update_iter = 0

    def set_parameters(self, parent_obj):
        """Assign sampled values to attributes on parent_obj."""
        for i, name in enumerate(self.param_names):
            val = self.throwing_values[:, i].unsqueeze(-1)
            setattr(parent_obj, name, val)
            # print(f"[NoCurriculum] Set {name} on parent_obj.")

    def update(self, parent_obj, past_rewards, past_avg_rewards):
        """
        At every call, re-sample all parameters uniformly and apply.
        """
        self.throwing_values = torch.rand(self.num_envs, len(self.param_names), device=self.device)
        for i in range(len(self.param_names)):
            low, high = self.param_ranges[i][0], self.param_ranges[i][1]
            self.throwing_values[:, i] = self.throwing_values[:, i] * (high - low) + low

        self.set_parameters(parent_obj)
        self.last_update_iter = len(past_avg_rewards[0]) if len(past_avg_rewards) > 0 else 0
