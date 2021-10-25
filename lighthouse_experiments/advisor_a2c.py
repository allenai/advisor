from advisor_losses import AdvisorWeightedStage
from allenact.utils.experiment_utils import PipelineStage
from projects.advisor.lighthouse_experiments.advisor_ppo import LightHouseAdvisorPPO


class LightHouseAdvisorA2C(LightHouseAdvisorPPO):
    """A2C and Imitation with adaptive reweighting."""

    def tag(self):
        return "LightHouseAdvisorA2C"

    def training_pipeline(self, **kwargs):
        alpha = 20
        training_steps = self.exp_params.TOTAL_TRAIN_STEPS
        a2c_info = self.rl_loss_default("a2c", steps=training_steps)

        return self._training_pipeline(
            named_losses={
                "advisor_loss": AdvisorWeightedStage(
                    rl_loss=a2c_info["loss"], fixed_alpha=alpha, fixed_bound=0
                ),
            },
            pipeline_stages=[
                PipelineStage(
                    loss_names=["advisor_loss"],
                    early_stopping_criterion=self.get_early_stopping_criterion(),
                    max_stage_steps=training_steps,
                ),
            ],
            num_mini_batch=a2c_info["num_mini_batch"],
            update_repeats=a2c_info["update_repeats"],
        )
