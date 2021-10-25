from typing import cast

import gym
from torch import nn

from advisor_losses import GAILDiscriminatorLoss, GAILPPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.sensor import SensorSuite
from allenact.embodiedai.models.basic_models import LinearActorCritic
from allenact.utils.experiment_utils import PipelineStage
from allenact_plugins.lighthouse_plugin.lighthouse_models import (
    LinearAdvisorActorCritic,
)
from allenact_plugins.minigrid_plugin.minigrid_sensors import EgocentricMiniGridSensor
from gail_models import MiniGridSimpleConvRNNWithDiscriminator
from poisoneddoors_plugin.poisoneddoors_models import (
    RNNActorCriticWithEmbedAndDiscriminator,
)
from projects.advisor.minigrid_and_pd_experiments.base import BaseExperimentConfig


class MgPdGAIL(BaseExperimentConfig):
    """Training with adaptive reweighing."""

    USE_EXPERT = False
    ALSO_USING_PPO = True

    def create_model(self, **kwargs) -> nn.Module:
        sensors = self.get_sensors()
        task_info = self.task_info()
        if task_info["name"] == "PoisonedDoors":
            return RNNActorCriticWithEmbedAndDiscriminator(
                input_uuid=sensors[0].uuid,
                num_embeddings=4,
                embedding_dim=128,
                input_len=1,
                action_space=gym.spaces.Discrete(
                    3 + task_info["env_info"]["num_doors"]
                ),
                observation_space=SensorSuite(sensors).observation_spaces,
                rnn_type=self.exp_params.RNN_TYPE,
                head_type=LinearActorCritic
                if not self.exp_params.INCLUDE_AUXILIARY_HEAD
                else Builder(  # type: ignore
                    LinearAdvisorActorCritic,
                    kwargs={
                        "ensure_same_init_aux_weights": self.exp_params.SAME_INIT_VALS_FOR_ADVISOR_HEAD
                    },
                ),
            )
        else:
            # Model for MiniGrid tasks
            return MiniGridSimpleConvRNNWithDiscriminator(
                action_space=gym.spaces.Discrete(
                    len(task_info["task_class"].class_action_names())
                ),
                num_objects=cast(EgocentricMiniGridSensor, sensors[0]).num_objects,
                num_colors=cast(EgocentricMiniGridSensor, sensors[0]).num_colors,
                num_states=cast(EgocentricMiniGridSensor, sensors[0]).num_states,
                observation_space=SensorSuite(sensors).observation_spaces,
                hidden_size=128,
                rnn_type=self.exp_params.RNN_TYPE,
                head_type=LinearActorCritic
                if not self.exp_params.INCLUDE_AUXILIARY_HEAD
                else Builder(  # type: ignore
                    LinearAdvisorActorCritic,
                    kwargs={
                        "ensure_same_init_aux_weights": self.exp_params.SAME_INIT_VALS_FOR_ADVISOR_HEAD
                    },
                ),
                recurrent_discriminator=True,
            )

    def extra_tag(self):
        return f"GAIL__lr_{self.exp_params.LR}"

    def training_pipeline(self, **kwargs):
        training_steps = self.total_train_steps()
        offpolicy_demo_info = self.offpolicy_demo_defaults(
            also_using_ppo=False  # We are but don't say so as this would reduce update repeats.
        )

        ppo_defaults = self.rl_loss_default("ppo", 1)

        gamma = 0.99
        use_gae = True
        gae_lambda = 1.0

        assert ppo_defaults["update_repeats"] % 2 == 0
        ppo_update_repeats = ppo_defaults["update_repeats"]
        gail_update_repeats = 5  # Default from ikostrikov
        gail_warmup_update_repeats = 100  # Default from ikostrikov

        gail_warmup_training_steps = min(
            training_steps,
            10 * (self.exp_params.NUM_TRAIN_SAMPLERS * self.exp_params.ROLLOUT_STEPS),
        )
        assert (
            gail_warmup_training_steps <= training_steps // 10
        )  # Don't spend more than 10% of training on warmup
        after_warmup_training_steps = training_steps - gail_warmup_training_steps

        return self._training_pipeline(
            named_losses={
                "gail_discriminator_loss": GAILDiscriminatorLoss(
                    data_iterator_builder=offpolicy_demo_info["data_iterator_builder"],
                    discriminator_observation_uuid="poisoned_door_state"
                    if self.task_name == "PoisonedDoors"
                    else "minigrid_ego_image",
                ),
                "gail_ppo_loss": GAILPPO(
                    **{
                        **PPOConfig,
                        "gamma": gamma,
                        "use_gae": use_gae,
                        "gae_lambda": gae_lambda,
                        "nrollouts": self.exp_params.NUM_TRAIN_SAMPLERS
                        // ppo_defaults["num_mini_batch"],
                        "rollout_len": self.exp_params.ROLLOUT_STEPS,
                    },
                ),
            },
            pipeline_stages=[
                PipelineStage(
                    loss_names=["gail_discriminator_loss"],  # Warmup
                    loss_update_repeats=[gail_warmup_update_repeats],
                    max_stage_steps=gail_warmup_training_steps,
                ),
                PipelineStage(
                    loss_names=["gail_discriminator_loss", "gail_ppo_loss"],
                    loss_update_repeats=[gail_update_repeats, ppo_update_repeats],
                    max_stage_steps=after_warmup_training_steps,
                    early_stopping_criterion=self.task_info().get(
                        "early_stopping_criterion"
                    ),
                ),
            ],
            num_mini_batch=ppo_defaults["num_mini_batch"],
            update_repeats=None,  # Specified in the pipeline stage
        )
