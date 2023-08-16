import gymnasium as gym
import numpy as np
from scipy.special import expit as sigmoid
from gymnasium import spaces


class HeatAlertEnv(gym.Env):
    """Class to simulate the environment for th online RL agent."""

    def __init__(
        self,
        posterior_coefficient_samples: dict[str, np.ndarray],
        baseline_states: dict[str, np.ndarray],
        effectiveness_states: dict[str, np.ndarray],
        budget_range: tuple[int, int],
        over_budget_penalty: float = 0.1,
        eval_mode: bool = False,
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

        Note: The code assumes that all posterior coefficients have the same number of samples.
            The number of samples will be determined by the first key in the posterior_coefficients
            dictionary. Similarly, it is assumed that the number of `episodes` is the same for
            all features. The number of features will be determined by the first key in the
            baseline_states dictionary.
        """
        super().__init__()

        self.baseline_keys = list(baseline_states.keys())
        self.effectiveness_keys = list(effectiveness_states.keys())
        self.baseline_dim = len(self.baseline_keys)
        self.effectiveness_dim = len(self.effectiveness_keys)
        self.budget_range = budget_range
        self.penalty = over_budget_penalty
        self.eval_mode = eval_mode

        self.posterior_coefficient_samples = posterior_coefficient_samples
        self.baseline_states = baseline_states
        self.effectiveness_states = effectiveness_states

        # deduce shapes (num days, num post samples, etc)
        coeffs_shape = next(iter(posterior_coefficient_samples.values())).shape
        feats_shape = next(iter(baseline_states.values())).shape
        self.n_posterior_samples = coeffs_shape[0]
        self.n_feature_episodes = feats_shape[0]
        self.n_days = feats_shape[1]

        # compute policy observation space; we will use:
        #   - baseline fixed features
        #   - covariate fixed features
        #   - number of alerts (2weeks)
        #   - alert lag
        #   - number of prev alerts in all episode
        # TODO: we could try a generalizing better the alert lags
        obs_dim = (
            self.baseline_dim
            + self.effectiveness_dim
            + 3  # 3 for number of alert variables
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)  # alert or no alert

    def reset(self):
        self.alert_buffer = []
        self.budget = np.random.randint(*self.budget_range)
        self.t = 0  # day of summer indicator
        self.feature_ep_index = np.random.choice(self.n_feature_episodes)
        return self.get_state()

    def over_budget(self):
        return sum(self.alert_buffer) > self.budget

    def get_state(self):
        baseline_feats = [
            self.baseline_states[k][self.feature_ep_index, self.t]
            for k in self.baseline_keys
        ]

        eff_feats = [
            self.effectiveness_states[k][self.feature_ep_index, self.t]
            for k in self.effectiveness_keys
        ]

        total_prev_alerts = sum(self.alert_buffer)
        prev_alerts_2wks = sum(self.alert_buffer[-14:])
        prev_alert_lag = 0 if len(self.alert_buffer) == 0 else self.alert_buffer[-1]
        alert_feats = [total_prev_alerts, prev_alerts_2wks, prev_alert_lag]

        return np.array(baseline_feats + eff_feats + alert_feats)

    def get_reward(self, posterior_index, action):
        baseline_contribs = [
            self.baseline_states[k][self.feature_ep_index, self.t]
            * self.posterior_coefficient_samples[k][posterior_index]
            for k in self.baseline_keys
        ]
        baseline = np.exp(sum(baseline_contribs))

        effectiveness_contribs = [
            self.effectiveness_states[k][self.feature_ep_index, self.t]
            * self.posterior_coefficient_samples[k][posterior_index]
            for k in self.effectiveness_keys
        ]
        effectiveness = sigmoid(sum(effectiveness_contribs))

        if self.over_budget():
            return 1 - baseline - self.penalty
        else:
            return 1 - baseline * (1 - effectiveness * action)

    def step(self, action: int):
        # advance state
        self.alert_buffer.append(action)
        self.t += 1
        new_state = self.get_state()

        # compute reward for the new state
        posterior_indices = (
            np.arange(self.n_posterior_samples)
            if self.eval_mode
            else [np.random.choice(self.n_posterior_samples)]
        )
        reward = np.mean([self.get_reward(i, action) for i in posterior_indices])
        done = self.t == self.n_days - 1

        # we can add more info here as needed, useful for callbacks, custom metrics
        info = {
            "episode_index": self.feature_ep_index,
            "budget": self.budget,
            "over_budget": self.over_budget(),
        }

        return new_state, reward, done, info


if __name__ == "__main__":
    # test

    n_posterior_samples = 100
    n_feature_episodes = 100
    n_days = 153
    n_baseline_feats = 10
    n_effectiveness_feats = 20
    baseline_keys = list("abc")
    effectiveness_keys = list("de")

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
        over_budget_penalty=0.1,
    )

    # step through a full episode until done with random actions
    obs = env.reset()
    done = False
    step = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        step += 1
        over_budget = info["over_budget"]
        print(
            f"{step}. action: {action}, reward: {reward:.2f}, done: {done}, over_budget: {over_budget}"
        )
