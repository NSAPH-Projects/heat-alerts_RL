import gymnasium as gym
import numpy as np
from scipy.special import expit as sigmoid
from gymnasium import spaces


class HeatAlertEnv(gym.Env):
    """Class to simulate the environment for the online RL agent."""

    def __init__(
        self,
        posterior_coefficient_samples: dict[str, np.ndarray],
        baseline_states: dict[str, np.ndarray],
        effectiveness_states: dict[str, np.ndarray],
        extra_states: dict[str, np.ndarray] = {},
        other_data: dict[str, np.ndarray] = {},
        penalty: float = 1.0,
        eval_mode: bool = False,
        sample_budget: bool = True,
        years = [],
        prev_alert_mean = 0,
        prev_alert_std = 1
    ):
        """Initialize the environment.

        Args:
            posterior_coefficients (dict[str, np.ndarray]): a dictionary where keys
                are the names of coefficients (baseline or effectiveness) and the
                values are 1-d arrays. The length of the each array is considered
                to be the number of posterior samples.
            baseline_states (dict[str, np.ndarray]): a dictionary where keys
                are the names of baseline features and the values are two-dimensional arrays.
                The first dimension is the number of episodes availables.
                Samples will be taking from the first dimension for each episode.
                The second dimension corresponds to the number of days of summer.
                Note that these states cannot contain information about alerts.
            effectiveness_states (np.ndarray): a dictionary where keys
                are the names of effectiveness features. The formatting is the same as with the
                baseline_states.
            budget_range (tuple[int, int]): A range to the allowed budget from. Each episode will
                sample uniformly from the interval [budget_range[0], budget_range[1])
            over_budget_penalty (float): penalty to apply when the agent tries to issue an alert
                but the budget is exceeded. Defaults to 0.1.
            eval_mode (bool): whether to run the environment in evaluation mode. In eval mode,
                the reward is averaged over all posterior coefficient samples instead using one sample.
            prev_alert_mean (float) and prev_alert_std (float): the mean and standard deviation of 
                the previous_alerts variable, to enable putting this variable on the same scale as the 
                rewards model training data.

        Note: The code assumes that all posterior coefficients have the same number of samples.
            The number of samples will be determined by the first key in the posterior_coefficients
            dictionary. Similarly, it is assumed that the number of `episodes` is the same for
            all features. The number of features will be determined by the first key in the
            baseline_states dictionary.
        """
        super().__init__()

        self.baseline_dim = len(baseline_states)
        self.extra_dim = len(extra_states)

        self.penalty = penalty
        self.eval_mode = eval_mode
        self.sample_budget = sample_budget
        self.years = years

        self.posterior_coefficient_samples = posterior_coefficient_samples
        self.baseline_states = baseline_states
        self.effectiveness_states = effectiveness_states
        self.extra_states = extra_states
        self.other_data = other_data

        self.prev_alert_mean = prev_alert_mean
        self.prev_alert_std = prev_alert_std

        # deduce shapes (num days, num post samples, etc)
        coeffs_shape = next(iter(posterior_coefficient_samples.values())).shape
        feats_shape = next(iter(baseline_states.values())).shape
        self.n_posterior_samples = coeffs_shape[0]
        self.n_feature_episodes = feats_shape[0]
        self.n_days = feats_shape[1]

        # compute policy observation space; we will use:
        #   - baseline fixed features
        #   - covariate fixed features
        #   - size of additional states
        #   - number of alerts (2weeks)
        #   - alert lag
        #   - number of prev alerts in all episode
        # TODO: we could try a generalizing better the alert lags
        obs_dim = (
            self.baseline_dim
            + self.extra_dim
            + 3  # alert variables
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)  # alert or no alert

    def reset(self, seed: int | None = None):
        # TODO: use a proper rng
        self.rng = np.random.default_rng(seed)
        self.attempted_alert_buffer = [] # what the RL tries to do
        self.allowed_alert_buffer = [] # what we allow the RL to do (based on the budget)
        self.t = 0  # day of summer indicator
        self.feature_ep_index = self.rng.choice(self.n_feature_episodes) #  We call this a hybrid environment because it uses a model for the rewards but samples the real weather trajectories for each summer.
        b = self.other_data["budget"][self.feature_ep_index, self.t]
        if self.sample_budget:
            self.budget = self.rng.integers(0.5*b, 1.5*b + 1)
        else:
            self.budget = b
        self.cum_reward = 0.0
        self.penalize = False
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        baseline_feats = [
            self.baseline_states[k][self.feature_ep_index, self.t]
            for k in self.baseline_states
        ]

        extra_feats = [
            self.extra_states[k][self.feature_ep_index, self.t]
            for k in self.extra_states
        ]

        total_prev_alerts = sum(self.allowed_alert_buffer)
        remaining_alerts = self.budget - total_prev_alerts
        prev_alerts_2wks = (sum(self.allowed_alert_buffer[-14:]) - self.prev_alert_mean)/(2 * self.prev_alert_std)
        prev_alert_lag = 0 if len(self.allowed_alert_buffer) == 0 else self.allowed_alert_buffer[-1]
        alert_feats = [remaining_alerts, prev_alerts_2wks, prev_alert_lag]

        return np.array(
            baseline_feats + extra_feats + alert_feats
        )

    def _get_reward(self, posterior_index, action, alert_feats):
        baseline_contribs = [
            self.baseline_states[k][self.feature_ep_index, self.t]
            * self.posterior_coefficient_samples[k][posterior_index]
            for k in self.baseline_states
        ]
        baseline = np.exp(sum(baseline_contribs) + 
                          # Note: remaining alerts is not a feature in the rewards model
                          alert_feats[1]*self.posterior_coefficient_samples["baseline_previous_alerts"][posterior_index] +
                          alert_feats[2]*self.posterior_coefficient_samples["baseline_alert_lag1"][posterior_index] + 
                          self.posterior_coefficient_samples["baseline_bias"][posterior_index])

        effectiveness_contribs = [
            self.effectiveness_states[k][self.feature_ep_index, self.t]
            * self.posterior_coefficient_samples[k][posterior_index]
            for k in self.effectiveness_states
        ]
        effectiveness = sigmoid(sum(effectiveness_contribs)  + 
                          # Note: remaining alerts is not a feature in the rewards model
                          alert_feats[1]*self.posterior_coefficient_samples["effectiveness_previous_alerts"][posterior_index] +
                          alert_feats[2]*self.posterior_coefficient_samples["effectiveness_alert_lag1"][posterior_index] +
                          self.posterior_coefficient_samples["effectiveness_bias"][posterior_index])

        if self.penalize:
            return 1 - baseline - self.penalty
        else:
            return 1 - baseline * (1 - effectiveness * action)

    def _get_info(self) -> dict:
        return {
            "episode_index": self.feature_ep_index,
            "budget": self.budget,
            "over_budget": self.penalize,
        }

    def step(self, action: int):
        # advance state
        self.t += 1
        new_state = self._get_obs()
        alert_feats = new_state[-3:]
        self.attempted_alert_buffer.append(action)
        if action == 1 and sum(self.allowed_alert_buffer) == self.budget:
            self.penalize = True
            action = 0
        else:
            self.penalize = False
        self.allowed_alert_buffer.append(action)

        # compute reward for the new state
        posterior_indices = (
            np.arange(self.n_posterior_samples)
            if self.eval_mode
            else [self.rng.choice(self.n_posterior_samples)]
        )
        reward = np.mean([self._get_reward(i, action, alert_feats) for i in posterior_indices])
        self.cum_reward += reward
        done = self.t == self.n_days - 1

        return new_state, reward, done, False, self._get_info()


if __name__ == "__main__":
    # test

    n_posterior_samples = 100
    n_feature_episodes = 100
    n_days = 153
    n_baseline_feats = 10
    n_effectiveness_feats = 20
    baseline_keys = list("abc")
    effectiveness_keys = list("de")

    np.random.seed(1234)

    posterior_coefficient_samples = {
        k: np.random.randn(n_posterior_samples)
        for k in baseline_keys + effectiveness_keys
    }
    baseline_fixed_features = {
        k: np.random.randn(n_feature_episodes, n_days) for k in baseline_keys
    }
    effectiveness_fixed_features = {
        k: np.random.randn(n_feature_episodes, n_days) for k in effectiveness_keys
    }
    env = HeatAlertEnv(
        posterior_coefficient_samples,
        baseline_fixed_features,
        effectiveness_fixed_features,
        budget_range=(10, 20),
        # penalty=0.1,
    )

    # step through a full episode until done with random actions
    obs = env.reset()
    done = False
    step = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        step += 1
        over_budget = info["over_budget"]
        print(
            f"{step}. action: {action}, reward: {reward:.2f}, done: {done}, over_budget: {over_budget}"
        )
