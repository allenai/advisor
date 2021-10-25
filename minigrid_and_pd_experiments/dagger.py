from allenact.utils.experiment_utils import PipelineStage, LinearDecay
from projects.advisor.minigrid_and_pd_experiments.base import BaseExperimentConfig


class MgPdDagger(BaseExperimentConfig):
    """Training with DAgger."""

    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name=task_name, USE_EXPERT=True, **kwargs)

    def extra_tag(self):
        return "Dagger__tf_{}__lr_{}".format(
            self.exp_params.TF_RATIO, self.exp_params.LR
        )

    def training_pipeline(self, **kwargs):
        training_steps = self.total_train_steps()
        steps_tf_stage = int(training_steps * self.exp_params.TF_RATIO)
        steps_bc_stage = training_steps - steps_tf_stage
        imitation_info = self.rl_loss_default("imitation")
        return self._training_pipeline(
            named_losses={"imitation_loss": imitation_info["loss"]},
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=steps_tf_stage,
                    teacher_forcing=LinearDecay(
                        startp=1.0, endp=0.0, steps=steps_tf_stage,
                    ),
                ),
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=steps_bc_stage,
                    early_stopping_criterion=self.task_info().get(
                        "early_stopping_criterion"
                    ),
                ),
            ],
            num_mini_batch=imitation_info["num_mini_batch"],
            update_repeats=imitation_info["update_repeats"],
        )
