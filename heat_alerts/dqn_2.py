
from typing import Optional, Sequence

from d3rlpy.algos.torch.dqn_impl import DQNImpl
from d3rlpy.algos.dqn import DQN

import torch
import numpy as np

from dqn_global import Pct90, HI_mean, HI_sd, device
# from heat_alerts.dqn_global import Pct90

class DQN_2Impl(DQNImpl):
    def compute_target(self, batch) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            next_actions = self._targ_q_func(batch.next_observations)
            action = next_actions.argmax(dim=1)
            original_targets = self._targ_q_func.compute_target(
                batch.next_observations,
                action,
                reduction="min", # reducing over an ensemble of Q functions
            )
            opposite_targets = self._targ_q_func.compute_target(
                batch.next_observations,
                torch.where(action==1, 0, 1), 
                reduction="min", # reducing over an ensemble of Q functions
            )
            if Pct90:
                quant_HI_county = torch.tensor(np.array([(b[6]*HI_sd + HI_mean) for b in batch.next_observations.cpu()])).to(device)
                targets = torch.where(torch.logical_and(action==1, quant_HI_county < 0.9), opposite_targets, original_targets) 
            else:
                targets = original_targets
            return targets


class DQN_2(DQN):
    _impl: Optional[DQN_2Impl] 
    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = DQN_2Impl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            n_critics=self._n_critics,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

class DoubleDQN_2Impl(DQNImpl):
    def compute_target(self, batch) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            action = self._predict_best_action(batch.next_observations)
            original_targets = self._targ_q_func.compute_target(
                batch.next_observations,
                action,
                reduction="min", # reducing over an ensemble of Q functions
            )
            opposite_targets = self._targ_q_func.compute_target(
                batch.next_observations,
                torch.where(action==1, 0, 1), 
                reduction="min", # reducing over an ensemble of Q functions
            )
            if Pct90:
                quant_HI_county = torch.tensor(np.array([(b[6]*HI_sd + HI_mean) for b in batch.next_observations.cpu()])).to(device)
                targets = torch.where(torch.logical_and(action==1, quant_HI_county < 0.9), opposite_targets, original_targets) 
            else:
                targets = original_targets
            return targets


class DoubleDQN_2(DQN):
    _impl: Optional[DoubleDQN_2Impl] 
    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = DoubleDQN_2Impl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            n_critics=self._n_critics,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()
