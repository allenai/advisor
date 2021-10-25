from advisor_losses import AdvisorWeightedStage
from allenact.utils.experiment_utils import PipelineStage
from projects.advisor.minigrid_and_pd_experiments.base import BaseExperimentConfig


class MgPdAdvisor(BaseExperimentConfig):
    """Training with adaptive reweighing."""

    def __init__(self, task_name: str, **kwargs):
        super().__init__(
            task_name=task_name,
            USE_EXPERT=True,
            SAME_INIT_VALS_FOR_ADVISOR_HEAD=False,
            INCLUDE_AUXILIARY_HEAD=True,
            **kwargs
        )

    def extra_tag(self):
        return "Advisor__alpha_{}__lr_{}".format(
            self.exp_params.FIXED_ALPHA, self.exp_params.LR,
        )

    def training_pipeline(self, **kwargs):
        training_steps = self.total_train_steps()
        ppo_info = self.rl_loss_default("ppo", steps=training_steps)
        alpha = self.exp_params.FIXED_ALPHA
        return self._training_pipeline(
            named_losses={
                "advisor_loss": AdvisorWeightedStage(
                    rl_loss=ppo_info["loss"], fixed_alpha=alpha, fixed_bound=0
                )
            },
            pipeline_stages=[
                PipelineStage(
                    loss_names=["advisor_loss"],
                    max_stage_steps=training_steps,
                    early_stopping_criterion=self.task_info().get(
                        "early_stopping_criterion"
                    ),
                ),
            ],
            num_mini_batch=ppo_info["num_mini_batch"],
            update_repeats=ppo_info["update_repeats"],
        )
