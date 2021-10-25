import typing
from typing import Dict, Union, Tuple, Iterator, Any
from typing import Optional

import numpy as np
import torch
from gym.utils import seeding

from advisor_losses import AlphaScheduler, AdvisorWeightedStage
from allenact.algorithms.offpolicy_sync.losses.abstract_offpolicy_loss import (
    AbstractOffPolicyLoss,
)
from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel
from allenact.base_abstractions.misc import Memory

_DATASET_CACHE: Dict[str, Any] = {}


class PoisonedDoorsOffPolicyExpertCELoss(AbstractOffPolicyLoss[ActorCriticModel]):
    def __init__(self, total_episodes_in_epoch: Optional[int] = None):
        super().__init__()
        self.total_episodes_in_epoch = total_episodes_in_epoch

    def loss(
        self,
        model: ActorCriticModel,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        memory: Memory,
        *args,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Dict[str, float], Memory, int]:

        rollout_len, nrollouts, _, = batch["poisoned_door_state"].shape

        observations = {}
        for k in ["poisoned_door_state"]:
            if k in batch:
                observations[k] = batch[k].view(
                    rollout_len, nrollouts, *batch[k].shape[2:]
                )

        ac_out, memory = model.forward(
            observations=observations,
            memory=memory,
            prev_actions=None,
            masks=batch["masks"],
        )

        expert_ce_loss = -ac_out.distributions.log_prob(
            batch["expert_action"].view(rollout_len, nrollouts, 1)
        ).mean()

        info = {"expert_ce": expert_ce_loss.item()}

        if self.total_episodes_in_epoch is not None:
            if "completed_episode_count" not in memory:
                memory["completed_episode_count"] = 0
            memory["completed_episode_count"] += (
                int(np.prod(batch["masks"].shape)) - batch["masks"].sum().item()
            )
            info["epoch_progress"] = (
                memory["completed_episode_count"] / self.total_episodes_in_epoch
            )

        return expert_ce_loss, info, memory, rollout_len * nrollouts


class PoisonedDoorsOffPolicyAdvisorLoss(AbstractOffPolicyLoss[ActorCriticModel]):
    def __init__(
        self,
        total_episodes_in_epoch: Optional[int] = None,
        fixed_alpha: Optional[float] = 1,
        fixed_bound: Optional[float] = 0.0,
        alpha_scheduler: AlphaScheduler = None,
        smooth_expert_weight_decay: Optional[float] = None,
        *args,
        **kwargs
    ):
        super().__init__()

        self.advisor_loss = AdvisorWeightedStage(
            rl_loss=None,
            fixed_alpha=fixed_alpha,
            fixed_bound=fixed_bound,
            alpha_scheduler=alpha_scheduler,
            smooth_expert_weight_decay=smooth_expert_weight_decay,
            *args,
            **kwargs
        )
        self.total_episodes_in_epoch = total_episodes_in_epoch

    def loss(
        self,
        step_count: int,
        model: ActorCriticModel,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        memory: Memory,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Dict[str, float], Memory, int]:

        rollout_len, nrollouts, _ = batch["poisoned_door_state"].shape

        observations = {"poisoned_door_state": batch["poisoned_door_state"]}

        ac_out, memory = model.forward(
            observations=observations,
            memory=memory,
            prev_actions=None,
            masks=batch["masks"].view(rollout_len, nrollouts, -1),
        )

        total_loss, losses_dict = self.advisor_loss.loss(
            step_count=step_count,
            batch={
                "observations": {
                    "expert_action": torch.cat(
                        (
                            batch["expert_action"].view(rollout_len, nrollouts, 1),
                            torch.ones(rollout_len, nrollouts, 1, dtype=torch.int64).to(
                                batch["expert_action"].device
                            ),
                        ),
                        dim=-1,
                    )
                }
            },
            actor_critic_output=ac_out,
        )

        info = {"offpolicy_" + key: val for key, val in losses_dict.items()}

        if self.total_episodes_in_epoch is not None:
            if "completed_episode_count" not in memory:
                memory["completed_episode_count"] = 0
            memory["completed_episode_count"] += (
                int(np.prod(batch["masks"].shape)) - batch["masks"].sum().item()
            )
            info["epoch_progress"] = (
                memory["completed_episode_count"] / self.total_episodes_in_epoch
            )

        return total_loss, info, memory, rollout_len * nrollouts


class PoisonedDoorsExpertTrajectoryIterator(Iterator):
    def __init__(
        self, num_doors: int, nrollouts: int, rollout_len: int, dataset_size: int,
    ):
        super(PoisonedDoorsExpertTrajectoryIterator, self).__init__()
        self.np_seeded_random_gen, _ = typing.cast(
            Tuple[np.random.RandomState, Any], seeding.np_random(0)
        )

        self.ndoors = num_doors
        self.nrollouts = nrollouts
        self.rollout_len = rollout_len
        self.dataset_size = dataset_size

        self.initial_observations = np.zeros(
            (rollout_len, nrollouts, 1), dtype=np.int64
        )

        self.mask = np.zeros((rollout_len, nrollouts, 1), dtype=np.float32)

        self.expert_actions = np.random.randint(
            4, 3 + num_doors, size=(self.dataset_size, 1)
        )

        self.current_ind = 0

    def __next__(self) -> Dict[str, torch.Tensor]:
        start = self.current_ind
        end = self.current_ind + self.nrollouts * self.rollout_len
        if end > self.dataset_size:
            raise StopIteration()
        self.current_ind = end

        return {
            "masks": torch.from_numpy(self.mask),
            "poisoned_door_state": torch.from_numpy(self.initial_observations),
            "expert_action": torch.from_numpy(
                self.expert_actions[start:end].reshape(
                    (self.rollout_len, self.nrollouts)
                )
            ),
        }


def create_poisoneddoors_offpolicy_data_iterator(
    num_doors: int, nrollouts: int, rollout_len: int, dataset_size: int,
) -> PoisonedDoorsExpertTrajectoryIterator:

    return PoisonedDoorsExpertTrajectoryIterator(
        num_doors=num_doors,
        nrollouts=nrollouts,
        rollout_len=rollout_len,
        dataset_size=dataset_size,
    )
