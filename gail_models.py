from typing import Callable, Dict, Optional, Any, cast

import gym
import numpy as np
import torch
from gym.spaces.dict import Dict as SpaceDict
from torch import nn, autograd

from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import Memory
from allenact.embodiedai.models.basic_models import LinearActorCritic, RNNStateEncoder
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.minigrid_plugin.minigrid_models import MiniGridSimpleConvRNN


class MiniGridDiscriminator(nn.Module):
    def __init__(
        self,
        observation_space: SpaceDict,
        num_objects: int,
        num_colors: int,
        num_states: int,
        num_actions: int,
        object_embedding_dim: int,
        action_embedding_dim: int,
        classifier_hidden_dim: int,
    ):
        super(MiniGridDiscriminator, self).__init__()

        self.object_embedding_dim = object_embedding_dim
        self.action_embedding_dim = action_embedding_dim

        # Same dimensionality used for colors and states
        self.color_embedding_dim = object_embedding_dim
        self.state_embedding_dim = object_embedding_dim

        # Input shapes
        vis_input_shape = observation_space["minigrid_ego_image"].shape
        agent_view_x, agent_view_y, view_channels = vis_input_shape
        assert agent_view_x == agent_view_y
        self.agent_view = agent_view_x
        self.view_channels = view_channels
        assert (np.array(vis_input_shape[:2]) >= 7).all(), (
            "MiniGridDiscriminator requires" "that the input size be at least 7x7."
        )

        # Object, Color, State --> Embeddings
        self.object_embedding = nn.Embedding(
            num_embeddings=num_objects, embedding_dim=self.object_embedding_dim
        )
        self.color_embedding = nn.Embedding(
            num_embeddings=num_colors, embedding_dim=self.color_embedding_dim
        )
        self.state_embedding = nn.Embedding(
            num_embeddings=num_states, embedding_dim=self.state_embedding_dim
        )
        # Same dimensionality used for actions
        self.action_embedding = nn.Embedding(
            num_embeddings=num_actions, embedding_dim=self.action_embedding_dim
        )

        # Classifier
        classifier_input_dim = (
            agent_view_x
            * agent_view_y
            * (
                self.object_embedding_dim
                + self.color_embedding_dim
                + self.state_embedding_dim
            )
            + self.action_embedding_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden_dim),
            nn.Tanh(),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim),
            nn.Tanh(),
            nn.Linear(classifier_hidden_dim, 1),
        )

        # self.returns = None
        # self.ret_rms = RunningMeanStd(shape=())

        self.train()

    def compute_grad_pen(
        self,
        expert_observations,
        expert_actions,
        policy_observations,
        policy_actions,
        pass_grad_through_encoder=False,
        expert_masks=None,
        policy_masks=None,
    ):
        alpha = torch.rand(*expert_observations["minigrid_ego_image"].shape[:2], 1)

        with torch.set_grad_enabled(pass_grad_through_encoder):
            encoded_expert_data = self.encode_minigrid_observations_actions(
                expert_observations, expert_actions, masks=expert_masks
            )
            encoded_policy_data = self.encode_minigrid_observations_actions(
                policy_observations, policy_actions, masks=policy_masks
            )
            alpha = alpha.expand_as(encoded_expert_data).to(encoded_expert_data.device)
            mixup_data = alpha * encoded_expert_data + (1 - alpha) * encoded_policy_data

        mixup_data.requires_grad = True
        disc = self.classifier(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = (
            (grad.norm(2, dim=1) - 1).pow(2).mean()
        )  # Scaling factor moved to the loss level
        return grad_pen

    def encode_minigrid_observations_actions(
        self, observations: Dict[str, Any], actions, masks: Optional[torch.Tensor],
    ):
        minigrid_ego_image = observations["minigrid_ego_image"]
        rollout_len, nrollouts, nrow, ncol, nchannels = minigrid_ego_image.shape

        minigrid_ego_image = minigrid_ego_image.view(
            rollout_len * nrollouts, nrow, ncol, nchannels
        )
        assert nrow == ncol == self.agent_view
        assert nchannels == self.view_channels == 3
        ego_object_embeds = self.object_embedding(minigrid_ego_image[:, :, :, 0].long())
        ego_color_embeds = self.color_embedding(minigrid_ego_image[:, :, :, 1].long())
        ego_state_embeds = self.state_embedding(minigrid_ego_image[:, :, :, 2].long())
        ego_embeds = torch.cat(
            (ego_object_embeds, ego_color_embeds, ego_state_embeds), dim=-1
        )
        action_embeds = self.action_embedding(actions.long())
        output_embeds = torch.cat(
            (
                ego_embeds.view(rollout_len, nrollouts, -1),
                action_embeds.view(rollout_len, nrollouts, -1),
            ),
            dim=-1,
        )
        return output_embeds

    def forward(
        self,
        observations: Dict[str, Any],
        actions,
        masks: Optional[torch.Tensor] = None,
    ):
        return self.classifier(
            self.encode_minigrid_observations_actions(
                observations=observations, actions=actions, masks=masks
            )
        )


class MiniGridDiscriminatorRNN(MiniGridDiscriminator):
    def __init__(
        self,
        observation_space: SpaceDict,
        num_objects: int,
        num_colors: int,
        num_states: int,
        num_actions: int,
        object_embedding_dim: int,
        action_embedding_dim: int,
        classifier_hidden_dim: int,
        rnn_output_dim: int = 256,
    ):
        super().__init__(
            **prepare_locals_for_super(
                {k: v for k, v in locals().items() if k != "rnn_output_dim"}
            )
        )

        # Classifier
        input_dim = (
            self.agent_view
            * self.agent_view
            * (
                self.object_embedding_dim
                + self.color_embedding_dim
                + self.state_embedding_dim
            )
            + self.action_embedding_dim
        )
        self.state_encoder = RNNStateEncoder(
            input_size=input_dim, hidden_size=rnn_output_dim
        )

        self.start_hidden_state = nn.Parameter(
            torch.zeros(self.state_encoder.num_recurrent_layers, 1, rnn_output_dim),
            requires_grad=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_dim, classifier_hidden_dim),
            nn.Tanh(),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim),
            nn.Tanh(),
            nn.Linear(classifier_hidden_dim, 1),
        )

        self.train()

    def encode_minigrid_observations_actions(
        self, observations: Dict[str, Any], actions, masks: Optional[torch.Tensor],
    ):
        minigrid_ego_image = observations["minigrid_ego_image"]
        rollout_len, nrollouts, nrow, ncol, nchannels = minigrid_ego_image.shape

        minigrid_ego_image = minigrid_ego_image.view(
            rollout_len * nrollouts, nrow, ncol, nchannels
        )
        assert nrow == ncol == self.agent_view
        assert nchannels == self.view_channels == 3
        ego_object_embeds = self.object_embedding(minigrid_ego_image[:, :, :, 0].long())
        ego_color_embeds = self.color_embedding(minigrid_ego_image[:, :, :, 1].long())
        ego_state_embeds = self.state_embedding(minigrid_ego_image[:, :, :, 2].long())
        ego_embeds = torch.cat(
            (ego_object_embeds, ego_color_embeds, ego_state_embeds), dim=-1
        )
        action_embeds = self.action_embedding(actions.long())
        output_embeds = torch.cat(
            (
                ego_embeds.view(rollout_len, nrollouts, -1),
                action_embeds.view(rollout_len, nrollouts, -1),
            ),
            dim=-1,
        )

        out, hidden = self.state_encoder.forward(
            x=cast(torch.FloatTensor, output_embeds),
            hidden_states=cast(
                torch.FloatTensor, self.start_hidden_state.repeat(1, nrollouts, 1)
            ),
            masks=cast(torch.FloatTensor, masks),
        )

        return out


class MiniGridSimpleConvRNNWithDiscriminator(MiniGridSimpleConvRNN):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        num_objects: int,
        num_colors: int,
        num_states: int,
        object_embedding_dim: int = 8,
        action_embedding_dim: int = 64,
        classifier_hidden_dim: int = 128,
        hidden_size=512,
        num_layers=1,
        rnn_type="GRU",
        head_type: Callable[
            ..., ActorCriticModel[CategoricalDistr]
        ] = LinearActorCritic,
        recurrent_discriminator: bool = False,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            num_objects=num_objects,
            num_colors=num_colors,
            num_states=num_states,
            object_embedding_dim=object_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
            head_type=head_type,
        )
        discriminator_class = (
            MiniGridDiscriminatorRNN
            if recurrent_discriminator
            else MiniGridDiscriminator
        )
        self.discriminator = discriminator_class(
            observation_space,
            num_objects,
            num_colors,
            num_states,
            action_space.n,
            object_embedding_dim,
            action_embedding_dim,
            classifier_hidden_dim,
        )

    def forward(
        self,
        observations: Dict[str, Any],
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ):
        out, memory = super(MiniGridSimpleConvRNNWithDiscriminator, self).forward(
            observations=observations,
            memory=memory,
            prev_actions=prev_actions,
            masks=masks,
        )
        out.extras["discriminator"] = self.discriminator
        return out, memory


class PoisonedDoorsDiscriminatorRNN(nn.Module):
    def __init__(
        self,
        input_uuid: str,
        num_action_embeddings: int,
        num_observation_embeddings: int,
        embedding_dim: int,
        classifier_hidden_dim: int,
        rnn_output_dim: int = 256,
    ):
        super().__init__()

        self.input_uuid = input_uuid
        self.observation_embedding = nn.Embedding(
            num_embeddings=num_observation_embeddings, embedding_dim=embedding_dim
        )
        self.action_embedding = nn.Embedding(
            num_embeddings=num_action_embeddings, embedding_dim=embedding_dim
        )

        # Classifier
        self.state_encoder = RNNStateEncoder(
            input_size=2 * embedding_dim, hidden_size=rnn_output_dim
        )

        self.start_hidden_state = nn.Parameter(
            torch.zeros(self.state_encoder.num_recurrent_layers, 1, rnn_output_dim),
            requires_grad=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_dim, classifier_hidden_dim),
            nn.Tanh(),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim),
            nn.Tanh(),
            nn.Linear(classifier_hidden_dim, 1),
        )

        self.train()

    def compute_grad_pen(
        self,
        expert_observations,
        expert_actions,
        policy_observations,
        policy_actions,
        pass_grad_through_encoder=False,
        expert_masks=None,
        policy_masks=None,
    ):
        alpha = torch.rand(*expert_observations[self.input_uuid].shape[:2], 1)

        with torch.set_grad_enabled(pass_grad_through_encoder):
            encoded_expert_data = self.encode_observations_and_actions(
                observations=expert_observations,
                actions=expert_actions,
                masks=expert_masks,
            )
            encoded_policy_data = self.encode_observations_and_actions(
                observations=policy_observations,
                actions=policy_actions,
                masks=policy_masks,
            )
            alpha = alpha.expand_as(encoded_expert_data).to(encoded_expert_data.device)
            mixup_data = alpha * encoded_expert_data + (1 - alpha) * encoded_policy_data

        mixup_data.requires_grad = True
        disc = self.classifier(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = (
            (grad.norm(2, dim=1) - 1).pow(2).mean()
        )  # Scaling factor moved to the loss level
        return grad_pen

    def encode_observations_and_actions(
        self, observations: Dict[str, Any], actions, masks: Optional[torch.Tensor],
    ):
        rollout_len, nrollouts = actions.shape
        obs_embed = self.observation_embedding(
            observations[self.input_uuid].view(rollout_len, nrollouts)
        )
        action_embed = self.action_embedding(actions)

        x = torch.cat((obs_embed, action_embed), dim=-1)
        assert len(x.shape) == 3

        out, hidden = self.state_encoder.forward(
            x=cast(torch.FloatTensor, x),
            hidden_states=cast(
                torch.FloatTensor, self.start_hidden_state.repeat(1, x.shape[1], 1)
            ),
            masks=cast(torch.FloatTensor, masks),
        )

        return out

    def forward(
        self,
        observations: Dict[str, Any],
        actions,
        masks: Optional[torch.Tensor] = None,
    ):
        return self.classifier(
            self.encode_observations_and_actions(
                observations=observations, actions=actions, masks=masks
            )
        )
