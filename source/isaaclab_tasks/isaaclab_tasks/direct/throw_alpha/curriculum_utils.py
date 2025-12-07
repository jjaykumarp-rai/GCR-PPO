import torch
import os


class Schedule:
    """
    Generic curriculum schedule.

    - param_names: list of attribute names on `parent_obj` that will be changed
    - param_ranges: shape [cl_levels, len(param_names)], each entry is a list/tuple:
        - [v]       -> fixed value for that level
        - [min,max] -> sampled uniformly every update() if random_sample=True
    - converg_crit: function(list_of_reward_histories) -> bool
    - cl_levels: number of curriculum stages
    - num_envs: number of envs
    - device: torch device
    """

    def __init__(self, param_names, param_ranges, converg_crit, cl_levels, num_envs, device, random_sample=True):
        # same assertion as your original
        assert len(param_ranges) and len(param_ranges[0]) == len(param_names), (
            "param_ranges must be [cl_levels, len(param_names)]."
        )
        self.param_names = param_names
        self.param_ranges = param_ranges
        self.converg_crit = converg_crit
        self.cl_levels = cl_levels
        self.current_level = 0
        self.last_update_iter = 0
        self.num_envs = num_envs
        self.device = device
        self.random_sample = random_sample  # if True, resample ranges every update()

    def set_parameters(self, parent_obj, use_print=True):
        # NOTE: values itself is not used outside; kept for compatibility
        self.values = torch.zeros(len(self.param_ranges[0]))
        for i, n in enumerate(self.param_names):
            range_i = self.param_ranges[self.current_level][i]
            # If the range is a single value, set it directly.
            if len(range_i) == 1:
                setattr(parent_obj, n, range_i[0])
                if use_print:
                    print(f"Set {n} in the curriculum to {range_i[0]}.")
            else:
                # Uniform sampling per-env
                low, high = range_i
                value = torch.rand(self.num_envs, 1, device=self.device) * (high - low) + low
                setattr(parent_obj, n, value)
                if use_print:
                    print(f"Set {n} in the curriculum to [{low}, {high}].")

    def update(self, parent_obj, past_rewards, past_avg_rewards):
        """
        parent_obj: usually the env or a wrapper with the attributes in param_names.
        past_avg_rewards: list-like of reward histories (like in your G1 training code).
        """

        # --- optional noise curriculum (same logic, but safe for Alpha) ---
        # Only triggers if:
        #   - enough iterations
        #   - (optionally) the task is "general"
        #   - noise_obs attribute exists
        if len(past_avg_rewards[0]) >= 4000:
            task_throw = getattr(getattr(parent_obj, "cfg", None), "env", None)
            task_throw = getattr(task_throw, "task_throw", "general")
            if (
                task_throw == "general"
                and hasattr(parent_obj, "noise_obs")
                and parent_obj.noise_obs < 1
                and self.last_update_iter < len(past_avg_rewards[0])
            ):
                new_noise_obs = min(1.0, parent_obj.noise_obs + 1 / 100)
                setattr(parent_obj, "noise_obs", new_noise_obs)
                print(f"Set noise_obs in the curriculum to {new_noise_obs}.")
                self.last_update_iter = len(past_avg_rewards[0])

        # --- main CL level advance ---
        if (
            len(past_avg_rewards[0]) > 0
            and self.converg_crit([x[self.last_update_iter:] for x in past_avg_rewards])
            and self.current_level < self.cl_levels - 1
        ):
            self.current_level += 1
            self.set_parameters(parent_obj)
            self.last_update_iter = len(past_avg_rewards[0])

        # --- optional continuous resampling within the same level ---
        elif self.random_sample:
            self.set_parameters(parent_obj, use_print=False)


class NoCurriculum:
    """
    Simple random "curriculum" (really just randomization within fixed ranges).

    - param_names: list of attribute names to set on parent_obj
    - param_ranges: [[min,max], ...] (one per param)
    """

    def __init__(self, param_names, param_ranges, num_envs, device):
        if os.path.exists("random_log.txt"):
            os.remove("random_log.txt")
        self.param_names = param_names
        self.param_ranges = param_ranges
        self.device = device
        self.num_envs = num_envs

        # Pre-sample a matrix of values [num_envs, num_params]
        self.throwing_values = torch.rand(self.num_envs, len(param_names), device=self.device)
        for i in range(len(param_names)):
            low, high = self.param_ranges[i]
            self.throwing_values[:, i] = self.throwing_values[:, i] * (high - low) + low
        self.last_update_iter = 0

    def set_parameters(self, parent_obj):
        for i, n in enumerate(self.param_names):
            setattr(parent_obj, n, self.throwing_values[:, i].unsqueeze(-1))

    def update(self, parent_obj, past_rewards, past_avg_rewards):
        # Resample every call
        self.throwing_values = torch.rand(self.num_envs, len(self.param_names), device=self.device)
        for i in range(len(self.param_names)):
            low, high = self.param_ranges[i]
            self.throwing_values[:, i] = self.throwing_values[:, i] * (high - low) + low
        self.set_parameters(parent_obj)
        self.last_update_iter = len(past_avg_rewards)
