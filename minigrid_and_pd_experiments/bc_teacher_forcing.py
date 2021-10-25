from allenact.utils.experiment_utils import PipelineStage, LinearDecay
from projects.advisor.minigrid_and_pd_experiments.base import BaseExperimentConfig


class MgPdBCTeacherForcing(BaseExperimentConfig):
    """Training with behavior cloning with teacher forcing of 1."""

    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name=task_name, USE_EXPERT=True, **kwargs)

    def extra_tag(self):
        return "BC_TEACHER_FORCING__lr_{}".format(self.exp_params.LR)

    def training_pipeline(self, **kwargs):
        training_steps = self.total_train_steps()
        loss_info = self.rl_loss_default("imitation")
        return self._training_pipeline(
            num_mini_batch=loss_info["num_mini_batch"],
            update_repeats=loss_info["update_repeats"],
            named_losses={"imitation_loss": loss_info["loss"]},
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=training_steps,
                    teacher_forcing=LinearDecay(
                        startp=1.0, endp=1.0, steps=training_steps,
                    ),
                ),
            ],
        )
