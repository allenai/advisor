from allenact.utils.experiment_utils import PipelineStage
from projects.advisor.minigrid_and_pd_experiments.base import BaseExperimentConfig


class MgPdPPO(BaseExperimentConfig):
    """Training with PPO."""

    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name=task_name, USE_EXPERT=False, **kwargs)

    def extra_tag(self):
        return "PPO__lr_{}".format(self.exp_params.LR)

    def training_pipeline(self, **kwargs):
        training_steps = self.total_train_steps()
        loss_info = self.rl_loss_default("ppo", steps=training_steps)
        return self._training_pipeline(
            named_losses={"ppo_loss": loss_info["loss"],},
            num_mini_batch=loss_info["num_mini_batch"],
            update_repeats=loss_info["update_repeats"],
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=training_steps,
                    early_stopping_criterion=self.task_info().get(
                        "early_stopping_criterion"
                    ),
                ),
            ],
        )
