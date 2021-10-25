from advisor_losses import MiniGridOffPolicyAdvisorLoss
from allenact.utils.experiment_utils import PipelineStage, OffPolicyPipelineComponent
from poisoneddoors_plugin.poisoneddoors_offpolicy import (
    PoisonedDoorsOffPolicyAdvisorLoss,
)
from projects.advisor.minigrid_and_pd_experiments.base import BaseExperimentConfig


class MgPdPPOWithOffPolicyAdvisor(BaseExperimentConfig):
    """PPO and Imitation with adaptive reweighting."""

    def __init__(self, task_name: str, **kwargs):
        super().__init__(
            task_name=task_name,
            SAME_INIT_VALS_FOR_ADVISOR_HEAD=False,
            INCLUDE_AUXILIARY_HEAD=True,
            **kwargs
        )

    def extra_tag(self):
        return "AdvisorOffPolicy__alpha_{}__lr_{}".format(
            self.exp_params.FIXED_ALPHA, self.exp_params.LR,
        )

    def training_pipeline(self, **kwargs):
        training_steps = self.total_train_steps()
        ppo_info = self.rl_loss_default("ppo", steps=training_steps)
        offpolicy_demo_info = self.offpolicy_demo_defaults(also_using_ppo=True)

        fixed_alpha = self.exp_params.FIXED_ALPHA
        assert fixed_alpha is not None

        if self.task_name == "PoisonedDoors":
            offpolicy_advisor_loss = PoisonedDoorsOffPolicyAdvisorLoss(
                fixed_alpha=fixed_alpha, fixed_bound=0
            )
        else:
            # MiniGrid Tasks
            offpolicy_advisor_loss = MiniGridOffPolicyAdvisorLoss(
                fixed_alpha=fixed_alpha, fixed_bound=0
            )

        return self._training_pipeline(
            named_losses={
                "ppo_loss": ppo_info["loss"],
                "offpolicy_advisor_loss": offpolicy_advisor_loss,
            },
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=training_steps,
                    early_stopping_criterion=self.task_info().get(
                        "early_stopping_criterion"
                    ),
                    offpolicy_component=OffPolicyPipelineComponent(
                        data_iterator_builder=offpolicy_demo_info[
                            "data_iterator_builder"
                        ],
                        loss_names=["offpolicy_advisor_loss"],
                        updates=offpolicy_demo_info["offpolicy_updates"],
                    ),
                ),
            ],
            num_mini_batch=offpolicy_demo_info["ppo_num_mini_batch"],
            update_repeats=offpolicy_demo_info["ppo_update_repeats"],
        )
