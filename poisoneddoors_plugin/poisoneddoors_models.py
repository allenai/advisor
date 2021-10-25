import typing
from typing import Dict, Tuple, Any, Union

import gym
import torch
import torch.nn as nn
from gym.spaces.dict import Dict as SpaceDict

from allenact.base_abstractions.misc import ActorCriticOutput, DistributionType, Memory
from allenact.embodiedai.models.basic_models import RNNActorCritic, LinearActorCritic
from allenact.utils.misc_utils import prepare_locals_for_super
from gail_models import PoisonedDoorsDiscriminatorRNN


class RNNActorCriticWithEmbed(RNNActorCritic):
    def __init__(
        self,
        input_uuid: str,
        num_embeddings: int,
        embedding_dim: int,
        input_len: int,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        num_layers: int = 1,
        rnn_type: str = "GRU",
        head_type=LinearActorCritic,
    ):
        hidden_size = embedding_dim * input_len
        super().__init__(
            input_uuid=input_uuid,
            action_space=action_space,
            observation_space=SpaceDict(
                {
                    input_uuid: gym.spaces.Box(
                        -float("inf"), float("inf"), shape=(hidden_size,)
                    )
                }
            ),
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
            head_type=head_type,
        )
        self.initial_embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )

    def forward(  # type: ignore
        self,
        observations: Dict[str, Union[torch.FloatTensor, Dict[str, Any]]],
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
        **kwargs,
    ) -> Tuple[ActorCriticOutput[DistributionType], Any]:

        input_obs = observations[self.input_uuid]
        obs = typing.cast(
            Dict[str, torch.FloatTensor],
            {
                self.input_uuid: self.initial_embedding(input_obs).view(
                    *input_obs.shape[:2], -1
                )
            },
        )
        return super(RNNActorCriticWithEmbed, self).forward(
            observations=obs, memory=memory, prev_actions=prev_actions, masks=masks,
        )


class RNNActorCriticWithEmbedAndDiscriminator(RNNActorCriticWithEmbed):
    def __init__(
        self,
        input_uuid: str,
        num_embeddings: int,
        embedding_dim: int,
        input_len: int,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        num_layers: int = 1,
        rnn_type: str = "GRU",
        head_type=LinearActorCritic,
    ):
        super(RNNActorCriticWithEmbedAndDiscriminator, self).__init__(
            **prepare_locals_for_super(locals())
        )

        self.discriminator = PoisonedDoorsDiscriminatorRNN(
            input_uuid=input_uuid,
            num_action_embeddings=action_space.n,
            num_observation_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            classifier_hidden_dim=128,
        )
        self.train()

    def forward(
        self,
        observations: Dict[str, Any],
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
        **kwargs,
    ):
        out, memory = super(RNNActorCriticWithEmbedAndDiscriminator, self).forward(
            observations=observations,
            memory=memory,
            prev_actions=prev_actions,
            masks=masks,
        )
        out.extras["discriminator"] = self.discriminator
        return out, memory
