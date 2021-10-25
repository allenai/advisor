from allenact.utils.experiment_utils import PipelineStage
from projects.advisor.lighthouse_experiments.base import BaseLightHouseExperimentConfig


class LightHouseBCAndPPO(BaseLightHouseExperimentConfig):
    """PPO and Imitation jointly."""

    def tag(self):
        return "LightHouseBCAndPPO"

    def training_pipeline(self, **kwargs):
        training_steps = self.exp_params.TOTAL_TRAIN_STEPS
        ppo_info = self.rl_loss_default("ppo", steps=training_steps)
        imitation_info = self.rl_loss_default("imitation")

        return self._training_pipeline(
            named_losses={
                "imitation_loss": imitation_info["loss"],
                "ppo_loss": ppo_info["loss"],
            },
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss", "ppo_loss"],
                    early_stopping_criterion=self.get_early_stopping_criterion(),
                    max_stage_steps=training_steps,
                ),
            ],
            num_mini_batch=min(
                info["num_mini_batch"] for info in [ppo_info, imitation_info]
            ),
            update_repeats=min(
                info["update_repeats"] for info in [ppo_info, imitation_info]
            ),
        )
