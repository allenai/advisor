from allenact.utils.experiment_utils import PipelineStage, OffPolicyPipelineComponent
from allenact_plugins.minigrid_plugin.minigrid_offpolicy import (
    MiniGridOffPolicyExpertCELoss,
)
from poisoneddoors_plugin.poisoneddoors_offpolicy import (
    PoisonedDoorsOffPolicyExpertCELoss,
)
from projects.advisor.minigrid_and_pd_experiments.base import BaseExperimentConfig


class MgPdPPOWithOffPolicyBC(BaseExperimentConfig):
    """PPO and off policy imitation."""

    def extra_tag(self):
        return "PPOAndOffPolicyBC__lr_{}".format(self.exp_params.LR)

    def training_pipeline(self, **kwargs):
        training_steps = self.total_train_steps()
        ppo_info = self.rl_loss_default("ppo", steps=training_steps)
        offpolicy_demo_info = self.offpolicy_demo_defaults(also_using_ppo=True)
        if self.task_name == "PoisonedDoors":
            offpolicy_expert_ce_loss = PoisonedDoorsOffPolicyExpertCELoss()
        else:
            # MiniGrid Tasks
            offpolicy_expert_ce_loss = MiniGridOffPolicyExpertCELoss()
        return self._training_pipeline(
            named_losses={
                "ppo_loss": ppo_info["loss"],
                "offpolicy_expert_ce_loss": offpolicy_expert_ce_loss,
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
                        loss_names=["offpolicy_expert_ce_loss"],
                        updates=offpolicy_demo_info["offpolicy_updates"],
                    ),
                ),
            ],
            num_mini_batch=offpolicy_demo_info["ppo_num_mini_batch"],
            update_repeats=offpolicy_demo_info["ppo_update_repeats"],
        )
