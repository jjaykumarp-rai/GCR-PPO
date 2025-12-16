import torch
import os


class Schedule:
    """Curriculum schedule helper for gradually adjusting environment parameters.

    This class manages *curriculum learning (CL)* over multiple levels/stages.
    At each level, it sets a collection of parameters on a `parent_obj`
    (typically an environment / task object) using either:
      - a fixed value (if the "range" is a single number), or
      - uniform random sampling within a specified range.

    The curriculum level advances when a user-provided convergence criterion
    indicates the agent has "converged enough" on the current difficulty.

    Key concepts:
    - `param_names`: names of attributes to set on the parent object.
    - `param_ranges`: per-level ranges for each parameter.
    - `converg_crit`: callable that decides when to advance levels.
    - `random_sample`: when True, re-sample ranged parameters even without
      changing curriculum level.

    IMPORTANT:
    - This class only *sets attributes* on `parent_obj`. It does not directly
      modify training, rewards, or simulation logic beyond those attributes.
    """

    def __init__(
        self,
        param_names,
        param_ranges,
        converg_crit,
        cl_levels,
        num_envs,
        device,
        random_sample=True,
    ):
        """Initialize the curriculum schedule.

        Args:
            param_names: List[str]. Names of parameters/attributes on `parent_obj`
                that will be updated by the curriculum.
            param_ranges: Per-level parameter specifications. Expected shape:
                [cl_levels][len(param_names)][1 or 2]
                - If inner list length == 1: treated as a fixed value.
                - If inner list length == 2: treated as [low, high] for uniform sampling.
            converg_crit: Callable. Function that takes in an array/list of reward histories
                (e.g., sliced reward streams) and returns True if curriculum should advance.
            cl_levels: int. Number of curriculum levels/stages.
            num_envs: int. Number of parallel environments (vectorized envs).
            device: torch device. Used for sampling tensors on CPU/GPU consistently.
            random_sample: bool. If True, parameters with ranges are resampled on every
                `update()` call (even if curriculum level does not advance).
        """
        # NOTE: Keeping original assertion behavior/intent; only clarifying it via comments.
        assert len(param_ranges), len(param_ranges[0]) == (cl_levels, len(param_names))

        self.param_names = param_names  # List[str] of variable names to modify on parent_obj
        self.param_ranges = param_ranges  # Per-level ranges/specs for each parameter
        self.converg_crit = converg_crit  # Callable that decides when to move to the next CL level
        self.cl_levels = cl_levels  # Total number of curriculum levels/stages

        # Track curriculum progression
        self.current_level = 0
        self.last_update_iter = 0  # Index into reward history at which the last update occurred

        # Sampling settings
        self.num_envs = num_envs
        self.device = device
        self.random_sample = random_sample  # Resample ranged params even without level increments

    def set_parameters(self, parent_obj, use_print=True):
        """Apply the current curriculum level's parameters onto `parent_obj`.

        For each parameter in `param_names`, we read its "spec" from
        `param_ranges[current_level][i]`:
        - If the spec contains a single value: set attribute to that constant.
        - If the spec contains two values [low, high]: sample uniformly per-env.

        Args:
            parent_obj: The object whose attributes should be updated (env/task instance).
            use_print: If True, print the chosen value/range for each parameter.
        """
        # NOTE: `self.values` is not used downstream in this snippet, but we preserve it
        # exactly as-is (no logic changes) since it might be referenced elsewhere.
        self.values = torch.zeros(len(self.param_ranges[0]))

        for i, n in enumerate(self.param_names):
            # Each parameter range spec is either [value] or [low, high]
            if len(self.param_ranges[self.current_level][i]) == 1:
                # Fixed value at this curriculum level
                setattr(parent_obj, n, self.param_ranges[self.current_level][i][0])
                if use_print:
                    print(
                        f"Set {n} in the curriculum to "
                        f"{self.param_ranges[self.current_level][i][0]}."
                    )
            else:
                # Uniform random sampling per env:
                # sample in [0,1) -> scale to [low, high]
                low = self.param_ranges[self.current_level][i][0]
                high = self.param_ranges[self.current_level][i][1]
                value = (
                    torch.rand(self.num_envs, 1, device=self.device) * (high - low) + low
                )
                setattr(parent_obj, n, value)
                if use_print:
                    print(f"Set {n} in the curriculum to [{low}, {high}].")

    def update(self, parent_obj, past_rewards, past_avg_rewards):
        """Update curriculum state and potentially re-apply parameter settings.

        This performs three responsibilities:
        1) A special-case schedule for `noise_obs` when task is "general".
        2) Check convergence; if converged, advance to the next curriculum level.
        3) If not advancing levels and `random_sample` is enabled, resample ranged params.

        Args:
            parent_obj: The object whose attributes should be updated (env/task instance).
            past_rewards: Reward history (not used in active code; kept for API compatibility).
            past_avg_rewards: Typically a list/tuple of reward streams, where index 0
                is used as a primary training progress signal in the logic below.
        """
        # The large block below is intentionally left commented out in the original code.
        # It appears to be an alternative mechanism for dynamically scaling rewards.
        # We keep it intact and only add this explanatory comment.
        '''# update stability coefficient
        if len(past_avg_rewards[0]) > 200:# and past_avg_rewards[1][-1] : # only start it after 200 iters
            original_scale = parent_obj.new_reward_scales[1]
            difference = parent_obj.desired_stability - past_avg_rewards[0][-1]
            #if difference > 0:
            #    difference /= 2 # slow down the increase
            stability_coef = min(0.1, max(parent_obj.reward_scales["stability"] + parent_obj.stability_adjustment_rate * difference, -1e-8))
            parent_obj.new_reward_scales[1] = float(stability_coef)
            rew_diff = past_avg_rewards[0][-1]*original_scale - stability_coef*past_avg_rewards[0][-1]
            parent_obj.new_reward_scales[0] += rew_diff / (-past_avg_rewards[1][-1]/parent_obj.MAX_DIST_PARAM + 1)
            #parent_obj.new_reward_scales[0] *= 1 + ((1-float(stability_coef))*past_avg_rewards[0][-1])/(past_avg_rewards[1][-1])    #(1-(original_scale-float(stability_coef)))*past_avg_rewards[0][-1] '''

        # ---------------------------------------------------------------------
        # 1) Special-case curriculum: gradually increase observation noise.
        # ---------------------------------------------------------------------
        # Conditions:
        # - Need at least 4000 points in the avg reward history.
        # - Only for "general" throwing task.
        # - Only update once per iteration index (guarded by last_update_iter check).
        # - noise_obs is incremented by 0.01 up to a cap of 1.0
        if len(past_avg_rewards[0]) >= 4000 and parent_obj.cfg.env.task_throw == "general":
            if (
                past_avg_rewards[0][-1] <= 1.0
                and parent_obj.noise_obs < 1
                and self.last_update_iter < len(past_avg_rewards[0])
            ):
                new_noise_obs = min(1.0, parent_obj.noise_obs + 1 / 100)
                setattr(parent_obj, "noise_obs", new_noise_obs)
                print(f"Set noise_obs in the curriculum to {new_noise_obs}.")
                self.last_update_iter = len(past_avg_rewards[0])

        # ---------------------------------------------------------------------
        # 2) Check convergence and advance curriculum level if needed.
        # ---------------------------------------------------------------------
        # - Only try to advance if we have any reward history at all.
        # - Slice reward history from `last_update_iter` onward so convergence is
        #   evaluated on the segment since the last curriculum change.
        # - Do not exceed the maximum curriculum level.
        if (
            len(past_avg_rewards[0]) > 0
            and self.converg_crit([x[self.last_update_iter:] for x in past_avg_rewards])
            and self.current_level < self.cl_levels - 1
        ):
            self.current_level += 1
            self.set_parameters(parent_obj)
            self.last_update_iter = len(past_avg_rewards[0])

        # ---------------------------------------------------------------------
        # 3) If we didn't advance levels, optionally resample parameters.
        # ---------------------------------------------------------------------
        elif self.random_sample:
            # Resample parameters that are within a range (silent by default).
            self.set_parameters(parent_obj, use_print=False)


class NoCurriculum:
    """Non-curriculum parameter randomizer.

    This class is effectively the "always random" alternative to `Schedule`.
    It samples each parameter uniformly within its maximum difficulty range
    and refreshes those values on every `update()` call.

    Notes:
    - It writes a file `random_log.txt` cleanup at init (if present).
    - It sets per-env values for each parameter (shape: [num_envs, 1]).
    """

    def __init__(self, param_names, param_ranges, num_envs, device):
        """Initialize the random parameter sampler.

        Args:
            param_names: List[str]. Names of attributes on `parent_obj` to randomize.
            param_ranges: List of [low, high] pairs for each parameter (max difficulty).
            num_envs: int. Number of parallel environments.
            device: torch device for tensor sampling.
        """
        # Remove old log file if it exists (kept as-is; might be part of the workflow).
        if os.path.exists("random_log.txt"):
            os.remove("random_log.txt")

        self.param_names = param_names
        self.param_ranges = param_ranges  # Assumes each entry is [low, high]
        self.device = device
        self.num_envs = num_envs

        # Pre-sample initial per-env parameter values.
        self.throwing_values = torch.rand(self.num_envs, len(param_names), device=self.device)
        for i in range(len(param_names)):
            low, high = param_ranges[i][0], param_ranges[i][1]
            self.throwing_values[:, i] = self.throwing_values[:, i] * (high - low) + low

        self.last_update_iter = 0  # Tracks how many avg reward samples have been seen

    def set_parameters(self, parent_obj):
        """Apply the currently sampled random parameters onto `parent_obj`.

        Each parameter is assigned a per-env tensor of shape [num_envs, 1].
        """
        for i, n in enumerate(self.param_names):
            setattr(parent_obj, n, self.throwing_values[:, i].unsqueeze(-1))
            # Printing intentionally omitted (kept consistent with original code).

    def update(self, parent_obj, past_rewards, past_avg_rewards):
        """Resample parameters and apply them to `parent_obj`.

        This method fully refreshes `throwing_values` every call and pushes the new
        values into the parent object.

        Args:
            parent_obj: The object whose attributes should be updated.
            past_rewards: Reward history (not used; kept for API compatibility).
            past_avg_rewards: Reward history used only to update `last_update_iter`.
        """
        # Resample per-env parameter values
        self.throwing_values = torch.rand(self.num_envs, len(self.param_names), device=self.device)
        for i in range(len(self.param_names)):
            low, high = self.param_ranges[i][0], self.param_ranges[i][1]
            self.throwing_values[:, i] = self.throwing_values[:, i] * (high - low) + low

        # Push sampled values into parent object
        self.set_parameters(parent_obj)

        # Track progress through reward history (kept as-is)
        self.last_update_iter = len(past_avg_rewards)
