
from typing import Optional, Sequence

from d3rlpy.algos.torch.dqn_impl import DQNImpl
from d3rlpy.algos.dqn import DQN

import torch
import numpy as np
# from torch_utility import TorchMiniBatch

class CPQImpl(DQNImpl):
    def compute_target(self, batch) -> torch.Tensor:
    # def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
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
                torch.where(action, 0, 1), 
                # torch.zeros(len(action).to(torch.int64)).to("cuda"), # can't do this because the function does one hot encoding and needs more than one action
                reduction="min", # reducing over an ensemble of Q functions
            )
            more_alerts = torch.tensor([b[13] for b in batch.next_observations]).to("cuda") # column of S with index 13 = "More_alerts"
            constrained_targets = torch.where(torch.logical_and(action, more_alerts == 0), opposite_targets, original_targets) 
            return constrained_targets


class CPQ(DQN):
    _impl: Optional[CPQImpl] 

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = CPQImpl(
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
