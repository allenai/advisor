from typing import Optional, Any

import gym
import numpy as np

from allenact.base_abstractions.sensor import Sensor
from allenact.utils.misc_utils import prepare_locals_for_super
from poisoneddoors_plugin.poisoneddoors_tasks import (
    PoisonedDoorsEnvironment,
    PoisonedDoorsTask,
    PoisonedEnvStates,
)


class PoisonedDoorCurrentStateSensor(
    Sensor[PoisonedDoorsEnvironment, PoisonedDoorsTask]
):
    def __init__(self, uuid: str = "poisoned_door_state", **kwargs: Any):
        self.nstates = len(PoisonedEnvStates)
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=0, high=self.nstates - 1, shape=(1,), dtype=int,)

    def get_observation(
        self,
        env: PoisonedDoorsEnvironment,
        task: Optional[PoisonedDoorsTask],
        *args,
        minigrid_output_obs: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> Any:
        return np.array([int(env.current_state.value)])
