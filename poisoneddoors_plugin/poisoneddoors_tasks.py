import random
from enum import Enum
from typing import Any, Tuple, Union, List, Optional, Dict

import gym
import numpy as np
from gym.utils import seeding

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import SensorSuite, Sensor
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.utils.experiment_utils import set_seed
from allenact.utils.system import get_logger


def get_combination(nactions: int, combination_length: int):
    s = random.getstate()
    random.seed(combination_length)
    comb = [random.randint(0, nactions - 1) for _ in range(combination_length)]
    random.setstate(s)
    return comb


class PoisonedEnvStates(Enum):
    choosing_door = 0
    entering_pass_start = 1
    entering_pass_cont = 2
    done = 3


class PoisonedDoorsEnvironment(object):
    def __init__(self, num_doors: int, combination_length: int):
        self.num_doors = num_doors
        self.combination_length = combination_length

        self.combination = get_combination(
            nactions=3, combination_length=self.combination_length
        )

        self.combination_index = 0
        self.max_comb_index = 0

        self.current_state = PoisonedEnvStates.choosing_door
        self.chosen_door: Optional[int] = None
        self.good_door_ind: Optional[int] = None

    @classmethod
    def class_action_names(cls, num_doors: int):
        return ("c0", "c1", "c2") + tuple(str(i) for i in range(num_doors))

    def action_names(self):
        return self.class_action_names(num_doors=self.num_doors)

    def reset(self, door_ind: int):
        assert 1 <= door_ind < self.num_doors
        self.good_door_ind = door_ind
        # print(self.good_door_ind)
        self.chosen_door = None
        self.current_state = PoisonedEnvStates.choosing_door
        self.combination_index = 0
        self.max_comb_index = 0

    def is_done(self):
        return self.current_state == PoisonedEnvStates.done

    def step(self, action: int) -> float:
        if action < 3 or self.current_state != self.current_state.choosing_door:
            if self.chosen_door is None:
                return 0.0
            else:
                assert self.chosen_door == 0, "Stepping when done."

                correct = self.combination[self.combination_index] == action

                if correct:
                    self.combination_index += 1
                    self.max_comb_index = max(
                        self.combination_index, self.max_comb_index
                    )
                else:
                    self.combination_index = 0

                if correct:
                    self.current_state = PoisonedEnvStates.entering_pass_cont
                elif not correct:
                    self.current_state = PoisonedEnvStates.done

                if self.combination_index >= len(self.combination):
                    self.current_state = PoisonedEnvStates.done
                    return 1.0
                return 0.0
        elif action == 3:
            self.chosen_door = 0
            self.combination_index = 0
            self.current_state = PoisonedEnvStates.entering_pass_start
            return 0.0
        else:
            self.current_state = PoisonedEnvStates.done
            self.chosen_door = action - 3
            return 2.0 * (1 if self.good_door_ind == action - 3 else -1)


class PoisonedDoorsTask(Task[PoisonedDoorsEnvironment]):
    """Defines an abstract embodied task in the light house gridworld.

    # Attributes

    env : The PoisonedDoorsEnvironment object.
    sensor_suite: Collection of sensors formed from the `sensors` argument in the initializer.
    task_info : Dictionary of (k, v) pairs defining task goals and other task information.
    max_steps : The maximum number of steps an agent can take an in the task before it is considered failed.
    observation_space: The observation space returned on each step from the sensors.
    """

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        pass

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(len(self.env.action_names()))

    def _step(self, action: int) -> RLStepResult:
        reward = self.env.step(action)

        return RLStepResult(
            observation=self.get_observations(),
            reward=reward,
            done=self.env.is_done(),
            info=None,
        )

    def reached_terminal_state(self) -> bool:
        return self.env.is_done()

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return PoisonedDoorsEnvironment.class_action_names(**kwargs)

    def action_names(self) -> Tuple[str, ...]:
        return self.env.action_names()

    def close(self) -> None:
        pass

    def query_expert(self, **kwargs) -> Tuple[Any, bool]:
        if self.env.current_state == PoisonedEnvStates.done:
            get_logger().warning("Trying to query expert with done task.")
            return (-1, False)
        elif self.env.current_state == PoisonedEnvStates.choosing_door:
            return (3 + self.env.good_door_ind, True)
        else:
            return (self.env.combination[self.env.combination_index], True)

    def metrics(self) -> Dict[str, Any]:
        metrics = super(PoisonedDoorsTask, self).metrics()

        for i in range(self.env.num_doors):
            metrics["chose_door_{}".format(i)] = 1.0 * (self.env.chosen_door == i)
        metrics["chose_no_door"] = 1.0 * (self.env.chosen_door is None)
        metrics["chose_good_door"] = self.env.chosen_door == self.env.good_door_ind

        metrics["opened_lock"] = 1.0 * (
            self.env.max_comb_index == self.env.combination_length
        )

        metrics["success"] = metrics["opened_lock"] or metrics["chose_good_door"]

        if self.env.chosen_door == 0:
            metrics["max_comb_correct"] = float(1.0 * self.env.max_comb_index)
        return metrics


class PoisonedDoorsTaskSampler(TaskSampler):
    def __init__(
        self,
        num_doors: int,
        combination_length: int,
        sensors: Union[SensorSuite, List[Sensor]],
        max_steps: int,
        max_tasks: Optional[int] = None,
        num_unique_seeds: Optional[int] = None,
        task_seeds_list: Optional[List[int]] = None,
        deterministic_sampling: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ):
        self.env = PoisonedDoorsEnvironment(
            num_doors=num_doors, combination_length=combination_length
        )

        self._last_sampled_task: Optional[PoisonedDoorsTask] = None
        self.sensors = (
            SensorSuite(sensors) if not isinstance(sensors, SensorSuite) else sensors
        )
        self.max_steps = max_steps
        self.max_tasks = max_tasks
        self.num_tasks_generated = 0
        self.deterministic_sampling = deterministic_sampling

        self.num_unique_seeds = num_unique_seeds
        self.task_seeds_list = task_seeds_list
        assert (self.num_unique_seeds is None) or (
            0 < self.num_unique_seeds
        ), "`num_unique_seeds` must be a positive integer."

        self.num_unique_seeds = num_unique_seeds
        self.task_seeds_list = task_seeds_list
        if self.task_seeds_list is not None:
            if self.num_unique_seeds is not None:
                assert self.num_unique_seeds == len(
                    self.task_seeds_list
                ), "`num_unique_seeds` must equal the length of `task_seeds_list` if both specified."
            self.num_unique_seeds = len(self.task_seeds_list)
        elif self.num_unique_seeds is not None:
            self.task_seeds_list = list(range(self.num_unique_seeds))

        assert (not deterministic_sampling) or (
            self.num_unique_seeds is not None
        ), "Cannot use deterministic sampling when `num_unique_seeds` is `None`."

        if (not deterministic_sampling) and self.max_tasks:
            get_logger().warning(
                "`deterministic_sampling` is `False` but you have specified `max_tasks < inf`,"
                " this might be a mistake when running testing."
            )

        self.seed: int = int(
            seed if seed is not None else np.random.randint(0, 2 ** 31 - 1)
        )
        self.np_seeded_random_gen: Optional[np.random.RandomState] = None
        self.set_seed(self.seed)

    @property
    def num_doors(self):
        return self.env.num_doors

    @property
    def combination_length(self):
        return self.env.combination_length

    @property
    def length(self) -> Union[int, float]:
        return (
            float("inf")
            if self.max_tasks is None
            else self.max_tasks - self.num_tasks_generated
        )

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        n = self.num_doors
        return n if self.num_unique_seeds is None else min(n, self.num_unique_seeds)

    @property
    def last_sampled_task(self) -> Optional[Task]:
        return self._last_sampled_task

    def next_task(self, force_advance_scene: bool = False) -> Optional[Task]:
        if self.length <= 0:
            return None

        if self.num_unique_seeds is not None:
            if self.deterministic_sampling:
                seed = self.task_seeds_list[
                    self.num_tasks_generated % len(self.task_seeds_list)
                ]
            else:
                seed = self.np_seeded_random_gen.choice(self.task_seeds_list)
        else:
            seed = self.np_seeded_random_gen.randint(0, 2 ** 31 - 1)

        self.num_tasks_generated += 1

        self.env.reset(door_ind=1 + (seed % (self.num_doors - 1)))
        return PoisonedDoorsTask(
            env=self.env, sensors=self.sensors, task_info={}, max_steps=self.max_steps
        )

    def close(self) -> None:
        pass

    @property
    def all_observation_spaces_equal(self) -> bool:
        return True

    def reset(self) -> None:
        self.num_tasks_generated = 0
        self.set_seed(seed=self.seed)

    def set_seed(self, seed: int) -> None:
        set_seed(seed)
        self.np_seeded_random_gen, _ = seeding.np_random(seed)
        self.seed = seed
