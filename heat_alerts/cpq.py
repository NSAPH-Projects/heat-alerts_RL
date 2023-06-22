
from typing import Optional, Sequence

from d3rlpy.algos.torch.dqn_impl import DQNImpl
from d3rlpy.algos.dqn import DQN

import torch
import numpy as np
# from torch_utility import TorchMiniBatch

from cpq_global import her, MA_mean, MA_sd, SA_mean, SA_sd, device
# boost = np.loadtxt('/n/dominici_nsaph_l3/Lab/projects/heat-alerts_mortality_RL/heat_alerts/cpq_boost.py')
# penalty = np.loadtxt('/n/dominici_nsaph_l3/Lab/projects/heat-alerts_mortality_RL/heat_alerts/cpq_penalty.py')
# boost = torch.FloatTensor(boost).to("cuda")
# penalty = torch.FloatTensor(penalty).to("cuda")

class CPQImpl(DQNImpl):
    def compute_target(self, batch) -> torch.Tensor:
    # def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        # print(batch.observations[0:3])
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
                # torch.zeros(len(action).to(torch.int64)).to(device), # can't do this because the function does one hot encoding and needs more than one action
                reduction="min", # reducing over an ensemble of Q functions
            )
            more_alerts = np.array([b[6]*MA_sd + MA_mean for b in batch.next_observations.cpu()]) # column of medium and small Ss
            # if her:
            #     p = np.round(1/(more_alerts + 1), 6) # prob of being at budget already after uniform sampling
            #     more_alerts = np.random.binomial(1, 1-p)
            if her:
                already_issued = np.array([b[5]*SA_sd + SA_mean for b in batch.next_observations.cpu()]) # column of medium and small Ss
                new_budgets = np.random.randint(0, already_issued + more_alerts + 1) # upper end is round bracket
                more_alerts = (already_issued < new_budgets).astype(int)
            more_alerts = torch.tensor(more_alerts).to(device)
            constrained_targets = torch.where(torch.logical_and(action==1, more_alerts > 0), opposite_targets, original_targets) 
            # penalized_targets = torch.where(torch.logical_and(action==1, more_alerts < 0.5), original_targets - penalty, original_targets)
            # boosted_targets = torch.where(torch.logical_and(action==1, more_alerts > 0.5), original_targets + boost, original_targets)
            # constrained_targets = torch.where(more_alerts > 0.5, boosted_targets, penalized_targets)
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
