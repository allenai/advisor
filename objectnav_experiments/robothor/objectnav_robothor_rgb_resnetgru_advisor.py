from typing import Optional

from allenact.base_abstractions.sensor import ExpertActionSensor
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from objectnav_experiments.objectnav_mixin_advisor import ObjectNavMixInADVISORConfig
from objectnav_experiments.objectnav_mixin_resnetgru_with_aux_head import (
    ObjectNavMixInResNetGRUWithAuxHeadConfig,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)


class ObjectNavRoboThorRGBAdvisor(
    ObjectNavRoboThorBaseConfig,
    ObjectNavMixInADVISORConfig,
    ObjectNavMixInResNetGRUWithAuxHeadConfig,
):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    SENSORS = [
        RGBSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(
            object_types=ObjectNavRoboThorBaseConfig.TARGET_TYPES,
        ),
        ExpertActionSensor(nactions=len(ObjectNavTask.class_action_names()),),
    ]

    @property
    def beta(self):
        if self.FIXED_BETA is None:
            return super(ObjectNavRoboThorRGBAdvisor, self).beta
        else:
            return self.FIXED_BETA

    def __init__(
        self,
        FIXED_ALPHA: Optional[float] = None,
        FIXED_BETA: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.FIXED_ALPHA = FIXED_ALPHA
        self.FIXED_BETA = FIXED_BETA

    @property
    def alpha(self):
        if self.FIXED_ALPHA is None:
            raise RuntimeError(
                "`FIXED_ALPHA` is `None`,"
                " this is fine for testing but should not occur"
                " if you wish to use this alpha value (e.g. in testing)."
            )
        return self.FIXED_ALPHA

    def tag(self):
        return f"Objectnav-RoboTHOR-RGB-ResNetGRU-ADVISOR_{self.alpha}_{self.beta}"
