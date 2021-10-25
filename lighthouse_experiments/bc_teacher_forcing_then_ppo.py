from allenact.utils.experiment_utils import PipelineStage, LinearDecay
from projects.advisor.lighthouse_experiments.base import BaseLightHouseExperimentConfig


class LightHouseBCTeacherForcingThenPPO(BaseLightHouseExperimentConfig):
    """Find goal in lighthouse env using imitation learning.

    Training with Imitation.
    """

    def tag(self):
        return "LightHouseBCTeacherForcingThenPPO"

    def training_pipeline(self, **kwargs):
        training_steps = self.exp_params.TOTAL_TRAIN_STEPS
        steps_per_pipeline_stage = training_steps // 2

        ppo_info = self.rl_loss_default("ppo", steps=steps_per_pipeline_stage)
        imitation_info = self.rl_loss_default("imitation")

        return self._training_pipeline(
            named_losses={
                "imitation_loss": imitation_info["loss"],
                "ppo_loss": ppo_info["loss"],
            },
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    teacher_forcing=LinearDecay(
                        startp=1.0, endp=1.0, steps=steps_per_pipeline_stage,
                    ),
                    max_stage_steps=steps_per_pipeline_stage,
                ),
                PipelineStage(
                    loss_names=["ppo_loss"],
                    early_stopping_criterion=self.get_early_stopping_criterion(),
                    max_stage_steps=steps_per_pipeline_stage,
                ),
            ],
            num_mini_batch=min(
                info["num_mini_batch"] for info in [ppo_info, imitation_info]
            ),
            update_repeats=min(
                info["update_repeats"] for info in [ppo_info, imitation_info]
            ),
        )
