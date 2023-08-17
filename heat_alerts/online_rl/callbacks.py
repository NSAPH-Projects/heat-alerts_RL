import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class AlertLoggingCallback(BaseCallback):
    """This callback logs in when the alerts are issued"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.when_alerted = []
        self.streaks = []
        self.current_streak = None
        self.last_alert = None
        self.num_over_budget = 0
        self.num_alerts = 0
        self.num_steps = 0
        self.rolled_rewards = 0.0

    def _on_step(self) -> bool:
        n_envs = len(self.training_env.envs)
        if self.current_streak is None:
            self.last_alert = np.zeros(n_envs, dtype=int)
            self.current_streak = np.zeros(n_envs, dtype=int)

        for i, env in enumerate(self.training_env.envs):
            self.num_steps += 1

            if env.over_budget():
                self.num_over_budget += 1

            if env.attempted_alert_buffer:
                prev_alert = self.last_alert[i]
                this_alert = env.attempted_alert_buffer[-1]
                if this_alert:  # alert issued
                    self.when_alerted.append(env.t)
                    self.num_alerts += 1
                    self.current_streak[i] += 1
                elif prev_alert:  # end streak
                    self.streaks.append(self.current_streak[i])
                    self.current_streak[i] = 0
                self.last_alert[i] = this_alert
            
            if env.done:
                self.rolled_rewards += env.cum_reward

        return True

    def _on_rollout_end(self):
        # Log the metrics to TensorBoard
        summary = {
            "average_training_rewards": self.rolled_rewards / len(self.training_env.envs),
            "over_budget_freq": self.num_over_budget / self.num_steps,
            "alerts_freq": self.num_alerts / self.num_steps,
            "average_t_alerts": np.mean(self.when_alerted) if self.when_alerted else 0,
            "stdev_t_alerts": np.std(self.when_alerted) if self.when_alerted else 0,
            "average_streak": np.mean(self.streaks) if self.streaks else 0,
            "stdev_streak": np.std(self.streaks) if self.streaks else 0,
        }

        for k, v in summary.items():
            self.logger.record(f"custom/{k}", v)

        # Reset counters
        self.when_alerted = []
        self.streaks = []
        self.current_streak = None
        self.last_alert = None
        self.num_over_budget = 0
        self.num_alerts = 0
        self.num_steps = 0