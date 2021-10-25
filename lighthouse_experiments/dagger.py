from allenact.utils.experiment_utils import PipelineStage, LinearDecay
from projects.advisor.lighthouse_experiments.base import BaseLightHouseExperimentConfig


class LightHouseDagger(BaseLightHouseExperimentConfig):
    """Find goal in lighthouse env using imitation learning.

    Training with Dagger.
    """

    def tag(self):
        return "LightHouseDagger"

    def training_pipeline(self, **kwargs):
        training_steps = self.exp_params.TOTAL_TRAIN_STEPS
        loss_info = self.rl_loss_default("imitation")
        return self._training_pipeline(
            named_losses={"imitation_loss": loss_info["loss"]},
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    teacher_forcing=LinearDecay(
                        startp=1.0, endp=0.0, steps=training_steps // 2,
                    ),
                    max_stage_steps=training_steps // 2,
                ),
                PipelineStage(
                    loss_names=["imitation_loss"],
                    early_stopping_criterion=self.get_early_stopping_criterion(),
                    max_stage_steps=training_steps // 2,
                ),
            ],
            num_mini_batch=loss_info["num_mini_batch"],
            update_repeats=loss_info["update_repeats"],
        )
