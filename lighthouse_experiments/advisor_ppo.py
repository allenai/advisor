from torch import nn

from advisor_losses import AdvisorWeightedStage
from allenact.base_abstractions.sensor import SensorSuite
from allenact.embodiedai.models.basic_models import RNNActorCritic
from allenact.utils.experiment_utils import PipelineStage
from allenact_plugins.lighthouse_plugin.lighthouse_models import (
    LinearAdvisorActorCritic,
)
from lighthouse_experiments.base import BaseLightHouseExperimentConfig


class LightHouseAdvisorPPO(BaseLightHouseExperimentConfig):
    """PPO and Imitation with adaptive reweighting."""

    def tag(self):
        return "LightHouseAdvisorPPO"

    def training_pipeline(self, **kwargs):
        alpha = 20
        training_steps = self.exp_params.TOTAL_TRAIN_STEPS
        ppo_info = self.rl_loss_default("ppo", steps=training_steps)

        return self._training_pipeline(
            named_losses={
                "advisor_loss": AdvisorWeightedStage(
                    rl_loss=ppo_info["loss"], fixed_alpha=alpha, fixed_bound=0
                )
            },
            pipeline_stages=[
                PipelineStage(
                    loss_names=["advisor_loss"],
                    early_stopping_criterion=self.get_early_stopping_criterion(),
                    max_stage_steps=training_steps,
                ),
            ],
            num_mini_batch=ppo_info["num_mini_batch"],
            update_repeats=ppo_info["update_repeats"],
        )

    def create_model(self, **kwargs) -> nn.Module:
        sensors = self.get_sensors()
        if self.exp_params.RECURRENT_MODEL:
            return RNNActorCritic(
                input_uuid=sensors[0].uuid,
                action_space=self._action_space(),
                observation_space=SensorSuite(sensors).observation_spaces,
                rnn_type="LSTM",
                head_type=Builder(  # type: ignore
                    LinearAdvisorActorCritic,
                    kwargs={"ensure_same_init_aux_weights": False},
                ),
            )
        else:
            return LinearAdvisorActorCritic(
                input_uuid=sensors[0].uuid,
                action_space=self._action_space(),
                observation_space=SensorSuite(sensors).observation_spaces,
                ensure_same_init_aux_weights=False,
            )
