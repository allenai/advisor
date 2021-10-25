from allenact.utils.experiment_utils import PipelineStage
from projects.advisor.minigrid_and_pd_experiments.base import BaseExperimentConfig


class MgPdBCThenPPO(BaseExperimentConfig):
    """Training with behavior cloning and then PPO."""

    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name=task_name, USE_EXPERT=True, **kwargs)

    def extra_tag(self):
        return "BCWithPPO__bc_{}__lr_{}".format(
            self.exp_params.TF_RATIO, self.exp_params.LR
        )

    def training_pipeline(self, **kwargs):
        training_steps = self.total_train_steps()

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
                    max_stage_steps=training_steps,
                    early_stopping_criterion=self.task_info().get(
                        "early_stopping_criterion"
                    ),
                )
            ],
            num_mini_batch=min(
                info["num_mini_batch"] for info in [ppo_info, imitation_info]
            ),
            update_repeats=min(
                info["update_repeats"] for info in [ppo_info, imitation_info]
            ),
        )
