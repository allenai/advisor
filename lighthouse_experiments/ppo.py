from allenact.utils.experiment_utils import PipelineStage
from projects.advisor.lighthouse_experiments.base import BaseLightHouseExperimentConfig


class LightHousePPO(BaseLightHouseExperimentConfig):
    """PPO only."""

    def tag(self):
        return "LightHousePPO"

    def training_pipeline(self, **kwargs):
        training_steps = self.exp_params.TOTAL_TRAIN_STEPS
        ppo_info = self.rl_loss_default("ppo", steps=training_steps)

        return self._training_pipeline(
            named_losses={"ppo_loss": ppo_info["loss"],},
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss"],
                    early_stopping_criterion=self.get_early_stopping_criterion(),
                    max_stage_steps=training_steps,
                ),
            ],
            num_mini_batch=ppo_info["num_mini_batch"],
            update_repeats=ppo_info["update_repeats"],
        )
