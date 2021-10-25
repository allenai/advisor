"""Defining the PPO loss for actor critic type models."""
import abc
import math
from typing import Dict, Union, Optional, Tuple, cast, Callable

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.running_mean_std import RunningMeanStd

from allenact.algorithms.offpolicy_sync.losses.abstract_offpolicy_loss import (
    AbstractOffPolicyLoss,
)
from allenact.algorithms.onpolicy_sync.losses import A2C, PPO
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
)
from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.utils.experiment_utils import Builder
from allenact.utils.tensor_utils import to_device_recursively
from gail_models import MiniGridDiscriminator


class AlphaScheduler(abc.ABC):
    def next(self, step_count: int, *args, **kwargs):
        raise NotImplementedError


class LinearAlphaScheduler(AlphaScheduler):
    def __init__(self, start: float, end: float, total_steps: int):
        self.start = start
        self.end = end
        self.total_steps = total_steps

    def next(self, step_count: int, *args, **kwargs):
        p = min(step_count / self.total_steps, 1)
        return self.start * (1.0 - p) + self.end * p


class AdvisorImitationStage(AbstractActorCriticLoss):
    """Implementation of the Advisor loss' stage 1 when main and auxiliary
    actors are equally weighted."""

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        # Imitation calculation
        observations = cast(Dict[str, torch.Tensor], batch["observations"])

        if "expert_action" not in observations:
            raise NotImplementedError(
                "AdvisorImitationStage loss requires that an `expert_action` is given as input"
            )

        expert_actions_and_mask = observations["expert_action"]

        assert expert_actions_and_mask.shape[-1] == 2
        expert_actions_and_mask_reshaped = expert_actions_and_mask.view(-1, 2)

        expert_actions = expert_actions_and_mask_reshaped[:, 0].view(
            *expert_actions_and_mask.shape[:-1], 1
        )
        expert_actions_masks = (
            expert_actions_and_mask_reshaped[:, 1]
            .float()
            .view(*expert_actions_and_mask.shape[:-1], 1)
        )

        expert_successes = expert_actions_masks.sum()
        should_report_loss = expert_successes.item() != 0

        main_expert_log_probs = actor_critic_output.distributions.log_prob(
            cast(torch.LongTensor, expert_actions)
        )
        aux_expert_log_probs = actor_critic_output.extras[
            "auxiliary_distributions"
        ].log_prob(cast(torch.LongTensor, expert_actions))

        assert main_expert_log_probs.shape == aux_expert_log_probs.shape
        assert (
            main_expert_log_probs.shape[: len(expert_actions_masks.shape)]
            == expert_actions_masks.shape
        )
        # Add dimensions to `expert_actions_masks` on the right to allow for masking
        # if necessary.
        len_diff = len(main_expert_log_probs.shape) - len(expert_actions_masks.shape)
        assert len_diff >= 0
        expert_actions_masks = expert_actions_masks.view(
            *expert_actions_masks.shape, *((1,) * len_diff)
        )

        aux_expert_ce_loss = -(
            expert_actions_masks * aux_expert_log_probs
        ).sum() / torch.clamp(expert_successes, min=1)

        main_expert_ce_loss = -(
            expert_actions_masks * main_expert_log_probs
        ).sum() / torch.clamp(expert_successes, min=1)

        total_loss = main_expert_ce_loss + aux_expert_ce_loss

        return (
            total_loss,
            {
                "main_expert_ce_loss": main_expert_ce_loss.item(),
                "aux_expert_ce_loss": aux_expert_ce_loss.item(),
                "total_loss": total_loss.item(),
            }
            if should_report_loss
            else {},
        )


class AdvisorWeightedStage(AbstractActorCriticLoss):
    """Implementation of the Advisor loss' second stage (simplest variant).

    # Attributes

    rl_loss: The RL loss to use, should be a loss object of type `PPO` or `A2C`
        (or a `Builder` that when called returns such a loss object).
    alpha : Exponent to use when reweighting the expert cross entropy loss.
        Larger alpha means an (exponentially) smaller weight assigned to the cross entropy
        loss. E.g. if a the weight with alpha=1 is 0.6 then with alpha=2 it is 0.6^2=0.36.
    bound : If the distance from the auxilary policy to expert policy is greater than
        this bound then the distance is set to 0.
    alpha_scheduler : An object of type `AlphaScheduler` which is before computing the loss
        in order to get a new value for `alpha`.
    smooth_expert_weight_decay : If not None, will redistribute (smooth) the weight assigned to the cross
        entropy loss at a particular step over the following `smooth_expert_steps` steps. Values
        of `smooth_expert_weight_decay` near 1 will increase how evenly weight is assigned
        to future steps. Values near 0 will decrease how evenly this weight is distributed
        with larger weight being given steps less far into the `future`.
        Here `smooth_expert_steps` is automatically defined from `smooth_expert_weight_decay` as detailed below.
    smooth_expert_steps : The number of "future" steps over which to distribute the current steps weight.
        This value is computed as `math.ceil(-math.log(1 + ((1 - r) / r) / 0.05) / math.log(r)) - 1` where
        `r=smooth_expert_weight_decay`. This ensures that the weight is always distributed over at least
        one additional step and that it is never distributed more than 20 steps into the future.
    """

    def __init__(
        self,
        rl_loss: Optional[Union[Union[PPO, A2C], Builder[Union[PPO, A2C]]]],
        fixed_alpha: Optional[float],
        fixed_bound: Optional[float],
        alpha_scheduler: AlphaScheduler = None,
        smooth_expert_weight_decay: Optional[float] = None,
        *args,
        **kwargs
    ):
        """Initializer.

        See the class documentation for parameter definitions not included below.

        fixed_alpha: This fixed value of `alpha` to use. This value is *IGNORED* if
            alpha_scheduler is not None.
        fixed_bound: This fixed value of the `bound` to use.
        """
        assert len(kwargs) == len(args) == 0

        super().__init__(*args, **kwargs)
        self.rl_loss: Union[PPO, A2C]
        if isinstance(rl_loss, Builder):
            self.rl_loss = rl_loss()
        else:
            self.rl_loss = rl_loss

        self.alpha = fixed_alpha
        self.bound = fixed_bound
        self.alpha_scheduler = alpha_scheduler
        self.smooth_expert_weight_decay = smooth_expert_weight_decay
        assert smooth_expert_weight_decay is None or (
            0 < smooth_expert_weight_decay < 1
        ), "`smooth_expert_weight_decay` must be between 0 and 1."
        if smooth_expert_weight_decay is not None:
            r = smooth_expert_weight_decay

            self.smooth_expert_steps = (
                math.ceil(-math.log(1 + ((1 - r) / r) / 0.05) / math.log(r)) - 1
            )

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        if self.alpha_scheduler is not None:
            self.alpha = self.alpha_scheduler.next(step_count=step_count)

        # Imitation calculation
        observations = cast(Dict[str, torch.Tensor], batch["observations"])
        if "expert_action" not in observations:
            raise NotImplementedError(
                "AdvisorImitationStage loss requires that an `expert_action` is given as input"
            )

        expert_actions_and_mask = observations["expert_action"]
        assert expert_actions_and_mask.shape[-1] == 2
        expert_actions_and_mask_reshaped = expert_actions_and_mask.view(-1, 2)

        expert_actions = expert_actions_and_mask_reshaped[:, 0].view(
            *expert_actions_and_mask.shape[:-1], 1
        )
        expert_actions_masks = (
            expert_actions_and_mask_reshaped[:, 1]
            .float()
            .view(*expert_actions_and_mask.shape[:-1], 1)
        )

        expert_successes = expert_actions_masks.sum()
        if expert_successes.item() == 0:
            return 0, {}

        main_expert_neg_cross_entropy = actor_critic_output.distributions.log_prob(
            cast(torch.LongTensor, expert_actions)
        )
        aux_expert_neg_cross_entropy = actor_critic_output.extras[
            "auxiliary_distributions"
        ].log_prob(cast(torch.LongTensor, expert_actions))

        # Add dimensions to `expert_actions_masks` on the right to allow for masking
        # if necessary.
        assert main_expert_neg_cross_entropy.shape == aux_expert_neg_cross_entropy.shape
        assert (
            main_expert_neg_cross_entropy.shape[: len(expert_actions_masks.shape)]
            == expert_actions_masks.shape
        )
        len_diff = len(main_expert_neg_cross_entropy.shape) - len(
            expert_actions_masks.shape
        )
        assert len_diff >= 0
        expert_actions_masks = expert_actions_masks.view(
            *expert_actions_masks.shape, *((1,) * len_diff)
        )

        aux_expert_ce_loss = -(
            expert_actions_masks * aux_expert_neg_cross_entropy
        ).sum() / torch.clamp(expert_successes, min=1)

        if self.bound > 0:
            top_bound = math.log(self.bound)
        else:
            top_bound = -float("inf")

        use_expert_weights = (
            torch.exp(self.alpha * aux_expert_neg_cross_entropy)
            * expert_actions_masks
            * (aux_expert_neg_cross_entropy >= top_bound).float()
        ).detach()

        if self.smooth_expert_weight_decay:
            # Here we smooth `use_expert_weights` so that a weight p assigned
            # to a step at time t is redisributed to steps
            # t, t+1, ..., t + self.smooth_expert_steps. This redistribution of
            # weight p is not allowed to pass from one episode to another and so
            # batch["masks"] must be used to prevent this.
            _, nsamplers, _ = expert_actions_masks.shape[1]

            start_shape = use_expert_weights.shape
            use_expert_weights = use_expert_weights.view(-1, nsamplers)

            padded_weights = F.pad(
                use_expert_weights, [0, 0, self.smooth_expert_steps, 0]
            )
            masks = cast(torch.Tensor, batch["masks"]).view(-1, nsamplers)
            padded_masks = F.pad(masks, [0, 0, self.smooth_expert_steps, 0])
            divisors = torch.ones_like(masks)  # Keep track of normalizing constants
            for i in range(1, self.smooth_expert_steps + 1):
                # Modify `use_expert_weights` so that weights are now computed as a
                # weighted sum of previous weights.
                masks = masks * padded_masks[self.smooth_expert_steps - i : -i, :]
                use_expert_weights += (
                    self.smooth_expert_weight_decay ** i
                ) * padded_weights[self.smooth_expert_steps - i : -i, :]
                divisors += masks * (self.smooth_expert_weight_decay ** i)
            use_expert_weights /= divisors
            use_expert_weights = use_expert_weights.view(*start_shape)

        # noinspection PyTypeChecker
        use_rl_weights = 1 - use_expert_weights

        weighted_main_expert_ce_loss = -(
            use_expert_weights * main_expert_neg_cross_entropy
        ).mean()

        total_loss = aux_expert_ce_loss + weighted_main_expert_ce_loss
        output_dict = {
            "aux_expert_ce_loss": aux_expert_ce_loss.item(),
            "weighted_main_expert_ce_loss": weighted_main_expert_ce_loss.item(),
            "non_zero_weight": (use_expert_weights > 0).float().mean().item(),
            "weight": use_expert_weights.mean().item(),
        }

        # RL Loss Computation
        if self.rl_loss is not None:
            rl_losses = self.rl_loss.loss_per_step(
                step_count=step_count,
                batch=batch,
                actor_critic_output=actor_critic_output,
            )
            if isinstance(rl_losses, tuple):
                rl_losses = rl_losses[0]

            action_loss, rl_action_loss_weight = rl_losses["action"]
            assert rl_action_loss_weight is None
            entropy_loss, rl_entropy_loss_weight = rl_losses["entropy"]

            def reweight(loss, w):
                return loss if w is None else loss * w

            weighted_action_loss = (
                use_rl_weights * (reweight(action_loss, rl_action_loss_weight))
            ).mean()

            weighted_entropy_loss = (
                use_rl_weights * reweight(entropy_loss, rl_entropy_loss_weight)
            ).mean()

            value_loss = rl_losses["value"][0].mean()
            total_loss += (
                (value_loss * rl_losses["value"][1])
                + weighted_action_loss
                + weighted_entropy_loss
            )
            output_dict.update(
                {
                    "value_loss": value_loss.item(),
                    "weighted_action_loss": weighted_action_loss.item(),
                    "entropy_loss": entropy_loss.mean().item(),
                }
            )

        output_dict["total_loss"] = total_loss.item()

        return total_loss, output_dict


class MiniGridOffPolicyAdvisorLoss(AbstractOffPolicyLoss[ActorCriticModel]):
    def __init__(
        self,
        fixed_alpha: Optional[float],
        fixed_bound: Optional[float],
        total_episodes_in_epoch: Optional[int] = None,
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
        *args,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Dict[str, float], Memory, int]:

        rollout_len, nrollouts = batch["minigrid_ego_image"].shape[:2]

        # Initialize Memory if empty
        if len(memory) == 0:
            spec = model.recurrent_memory_specification
            for key in spec:
                dims_template, dtype = spec[key]
                # get sampler_dim and all_dims from dims_template (and nrollouts)

                dim_names = [d[0] for d in dims_template]
                sampler_dim = dim_names.index("sampler")

                all_dims = [d[1] for d in dims_template]
                all_dims[sampler_dim] = nrollouts

                memory.check_append(
                    key=key,
                    tensor=torch.zeros(
                        *all_dims,
                        dtype=dtype,
                        device=cast(torch.Tensor, batch["minigrid_ego_image"]).device
                    ),
                    sampler_dim=sampler_dim,
                )

        # Forward data (through the actor and critic)
        ac_out, memory = model.forward(
            observations=batch,
            memory=memory,
            prev_actions=None,  # type:ignore
            masks=cast(torch.FloatTensor, batch["masks"]),
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


class GAILDiscriminatorLoss(AbstractActorCriticLoss):
    def __init__(
        self,
        data_iterator_builder,
        discriminator_observation_uuid: str,
        gradient_penalty_coeff: int = 10,
    ):
        super().__init__()
        self.data_iterator_builder = data_iterator_builder
        self.data_iterator = data_iterator_builder()
        self.discriminator_observation_uuid = discriminator_observation_uuid
        self.gradient_penalty_coeff = gradient_penalty_coeff

    def get_next_demo_batch(self):
        try:
            expert_batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = self.data_iterator_builder()
            expert_batch = next(self.data_iterator)
        return expert_batch

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        expert_batch = cast(Dict[str, torch.Tensor], self.get_next_demo_batch())
        device = batch["observations"][self.discriminator_observation_uuid].device
        expert_batch = to_device_recursively(expert_batch, device=device, inplace=True)

        rollout_len, nrollouts = expert_batch[
            self.discriminator_observation_uuid
        ].shape[:2]

        # expert_batch == offpolicy
        expert_actions = expert_batch["expert_action"]
        expert_masks = expert_batch["masks"]

        # batch == onpolicy
        policy_observations = cast(Dict[str, torch.Tensor], batch["observations"])
        policy_actions = batch["actions"]
        policy_masks = batch["masks"]

        assert (
            expert_batch[self.discriminator_observation_uuid].shape
            == policy_observations[self.discriminator_observation_uuid].shape
        )
        assert expert_actions.shape == policy_actions.shape
        assert expert_masks.shape == policy_masks.shape
        assert expert_actions.shape[:2] == (rollout_len, nrollouts)

        discriminator_network: MiniGridDiscriminator = actor_critic_output.extras[
            "discriminator"
        ]

        expert_logits = discriminator_network(
            observations=expert_batch, actions=expert_actions, masks=expert_masks
        )
        expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, torch.ones(expert_logits.size()).to(device)
        )

        policy_logits = discriminator_network(
            observations=policy_observations, actions=policy_actions, masks=policy_masks
        )
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_logits, torch.zeros(policy_logits.size()).to(device)
        )

        gradient_penalty = discriminator_network.compute_grad_pen(
            expert_observations=expert_batch,
            expert_actions=expert_actions,
            policy_observations=policy_observations,
            policy_actions=policy_actions,
            expert_masks=expert_masks,
            policy_masks=policy_masks,
        )

        return (
            expert_loss + policy_loss + self.gradient_penalty_coeff * gradient_penalty,
            {
                "gail_discriminator": (expert_loss + policy_loss).item(),
                "gail_gradient_penalty": gradient_penalty.item(),
            },
        )


def _compute_returns_and_adv(
    rewards, next_value, use_gae, gamma, tau, value_preds, masks
):
    returns = torch.zeros_like(value_preds)
    if use_gae:
        assert torch.all(torch.eq(value_preds[-1], next_value))
        gae = 0
        for step in reversed(range(rewards.size(0))):
            delta = (
                rewards[step]
                + gamma * value_preds[step + 1] * masks[step + 1]
                - value_preds[step]
            )
            gae = delta + gamma * tau * masks[step + 1] * gae
            returns[step] = gae + value_preds[step]
    else:
        returns[-1] = next_value
        for step in reversed(range(rewards.size(0))):
            returns[step] = returns[step + 1] * gamma * masks[step + 1] + rewards[step]

    advantages = returns[:-1] - value_preds[:-1]
    normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    return returns, normalized_advantages


class GAILPPO(AbstractActorCriticLoss):
    def __init__(
        self,
        clip_param: float,
        value_loss_coef: float,
        entropy_coef: float,
        gamma: float,
        use_gae: bool,
        gae_lambda: float,
        nrollouts: int,
        rollout_len: int,
        use_clipped_value_loss=True,
        clip_decay: Optional[Callable[[int], float]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_decay = clip_decay if clip_decay is not None else (lambda x: 1.0)
        self.gamma = gamma
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        self.running_means_std_of_returns = RunningMeanStd(shape=(1,))

        self.nrollouts = nrollouts
        self.rollout_len = rollout_len

    @staticmethod
    def _unflatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        """Given a tensor of size (t*n, ...) 'unflatten' it to size (t, n, ..).

        # Parameters
        t : first dimension of desired tensor.
        n : second dimension of desired tensor.
        tensor : target tensor to be unflattened.

        # Returns
        Unflattened tensor of size (t, n, ...)
        """
        return tensor.view(t, n, *tensor.size()[1:])

    def loss_per_step(
        self,
        step_count: int,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
    ) -> Dict[str, Tuple[torch.Tensor, Optional[float]]]:
        # Based on function with the same name in `PPO` (ppo.py)
        # Rewards are model based, hence, returns and advantages are recalculated.
        # Since next_value_pred of the N+1th observation isn't available, we reduce the time horizon
        # by one and calculate standard PPO losses. (Hence, the `[:-1]` are various places.

        actions = cast(torch.LongTensor, batch["actions"])

        # Calculate rewards
        observations = batch["observations"]
        discriminator = actor_critic_output.extras["discriminator"]
        unnorm_rewards = discriminator(
            observations=observations, actions=actions, masks=batch["masks"]
        ).detach()
        rewards = unnorm_rewards / (
            math.sqrt(float(self.running_means_std_of_returns.var[0])) + 1e-8
        )

        # computing returns expects data to be fed in a (rollout_len, nrollouts, 1) format
        # further reducing rewards' horizon by 1 so that batch_values is sufficient without needing
        # the next predicted value (exposed only at the level of the engine).
        rewards = rewards[:-1]

        batch_values = batch["values"]
        batch_masks = batch["masks"]

        # computing returns and advantages based on model based reward predictions
        next_value = batch_values[-1]
        returns, norm_adv_targ = _compute_returns_and_adv(
            rewards=rewards,
            next_value=next_value,
            use_gae=self.use_gae,
            gamma=self.gamma,
            tau=self.gae_lambda,
            value_preds=batch_values,
            masks=batch_masks,
        )

        self.running_means_std_of_returns.update(returns.view(-1).cpu().numpy())

        # reducing time horizon by one
        values = actor_critic_output.values[:-1]
        dist_entropy = actor_critic_output.distributions.entropy()[:-1]
        action_log_probs = actor_critic_output.distributions.log_prob(actions)[:-1]
        batch_old_action_log_probs = batch["old_action_log_probs"][:-1]
        batch_values = batch_values[:-1]
        returns = returns[:-1]

        # Everything used next is (rollout_len - 1, nrollouts, 1)
        # action_log_probs
        # batch_old_action_log_probs
        # norm_adv_targ
        # values
        # batch_values
        # returns
        def add_trailing_dims(t: torch.Tensor):
            assert len(t.shape) <= len(batch["norm_adv_targ"].shape)
            return t.view(
                t.shape + ((1,) * (len(batch["norm_adv_targ"].shape) - len(t.shape)))
            )

        dist_entropy = add_trailing_dims(dist_entropy)

        clip_param = self.clip_param * self.clip_decay(step_count)

        # Standard PPO loss components (but based on model based rewards instead of env based ones)
        ratio = torch.exp(action_log_probs - batch_old_action_log_probs)
        ratio = add_trailing_dims(ratio)

        surr1 = ratio * norm_adv_targ
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * norm_adv_targ
        action_loss = -torch.min(surr1, surr2)

        if self.use_clipped_value_loss:
            value_pred_clipped = batch_values + (values - batch_values).clamp(
                -clip_param, clip_param
            )
            value_losses = (values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (cast(torch.FloatTensor, returns) - values).pow(2)

        # noinspection PyUnresolvedReferences
        assert (
            value_loss.shape
            == action_loss.shape
            == value_loss.shape
            == (self.rollout_len - 1, self.nrollouts, 1)
        )
        return {
            "value": (value_loss, self.value_loss_coef),
            "action": (action_loss, None),
            "entropy": (dist_entropy.mul_(-1.0), self.entropy_coef),  # type: ignore
        }

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        # Same as `loss` in `PPO` (ppo.py)
        losses_per_step = self.loss_per_step(
            step_count=step_count, batch=batch, actor_critic_output=actor_critic_output,
        )
        if isinstance(losses_per_step[0], tuple):
            losses_per_step = losses_per_step[0]

        losses = {
            key: (loss.mean(), weight)
            for (key, (loss, weight)) in losses_per_step.items()
        }

        total_loss = sum(
            loss * weight if weight is not None else loss
            for loss, weight in losses.values()
        )

        return (
            total_loss,
            {**{key: loss.item() for key, (loss, _) in losses.items()},},
        )
