from advisor_losses import (
    AdvisorImitationStage,
    AdvisorWeightedStage,
)
from allenact.utils.experiment_utils import PipelineStage, LinearDecay
from projects.advisor.minigrid_and_pd_experiments.base import BaseExperimentConfig


class MgPdDaggerThenAdvisor(BaseExperimentConfig):
    """Training with DAgger followed adaptive reweighing."""

    def __init__(self, task_name: str, **kwargs):
        super().__init__(
            task_name=task_name,
            USE_EXPERT=True,
            SAME_INIT_VALS_FOR_ADVISOR_HEAD=False,
            INCLUDE_AUXILIARY_HEAD=True,
            **kwargs
        )

    def extra_tag(self):
        return "DaggerThenAdvisor__alpha_{}__lr_{}__tf_{}".format(
            self.exp_params.FIXED_ALPHA, self.exp_params.LR, self.exp_params.TF_RATIO,
        )

    def training_pipeline(self, **kwargs):
        training_steps = self.total_train_steps()
        steps_advisor_warmup_stage = int(training_steps * self.exp_params.TF_RATIO)
        steps_advisor_weighted_stage = training_steps - steps_advisor_warmup_stage

        ppo_info = self.rl_loss_default("ppo", steps=steps_advisor_weighted_stage)
        fixed_alpha = self.exp_params.FIXED_ALPHA
        return self._training_pipeline(
            named_losses={
                "advisor_imitation_warmup": AdvisorImitationStage(),
                "advisor_loss": AdvisorWeightedStage(
                    rl_loss=ppo_info["loss"], fixed_alpha=fixed_alpha, fixed_bound=0,
                ),
            },
            pipeline_stages=[
                PipelineStage(
                    loss_names=["advisor_imitation_warmup"],
                    max_stage_steps=steps_advisor_warmup_stage,
                    teacher_forcing=LinearDecay(
                        startp=1.0, endp=0.0, steps=steps_advisor_warmup_stage,
                    ),
                ),
                PipelineStage(
                    loss_names=["advisor_loss"],
                    max_stage_steps=steps_advisor_weighted_stage,
                    early_stopping_criterion=self.task_info().get(
                        "early_stopping_criterion"
                    ),
                ),
            ],
            num_mini_batch=ppo_info["num_mini_batch"],
            update_repeats=ppo_info["update_repeats"],
        )
