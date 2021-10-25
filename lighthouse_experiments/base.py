import math
from abc import ABC
from typing import Dict, Any, List, Optional, Tuple, Union, NamedTuple

import gym
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses import PPO, A2C
from allenact.algorithms.onpolicy_sync.losses.a2cacktr import A2CConfig
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.misc import Loss
from allenact.base_abstractions.sensor import (
    SensorSuite,
    Sensor,
    ExpertActionSensor,
)
from allenact.base_abstractions.task import TaskSampler
from allenact.embodiedai.models.basic_models import LinearActorCritic, RNNActorCritic
from allenact.utils.experiment_utils import (
    Builder,
    LinearDecay,
    PipelineStage,
    TrainingPipeline,
)
from allenact_plugins.lighthouse_plugin.lighthouse_environment import (
    LightHouseEnvironment,
)
from allenact_plugins.lighthouse_plugin.lighthouse_sensors import (
    FactorialDesignCornerSensor,
)
from allenact_plugins.lighthouse_plugin.lighthouse_tasks import (
    FindGoalLightHouseTaskSampler,
)
from allenact_plugins.lighthouse_plugin.lighthouse_util import StopIfNearOptimal


class LighthouseExperimentParams(NamedTuple):
    WORLD_DIM: int = 2
    VIEW_RADIUS: int = 1
    EXPERT_VIEW_RADIUS: int = 15
    WORLD_RADIUS: int = 15
    DEGREE: int = -1
    MAX_STEPS: int = 1000
    GPU_ID: Optional[int] = None
    NUM_TRAIN_SAMPLERS: int = 20 if torch.cuda.is_available() else 2
    NUM_TEST_TASKS: int = 200
    RECURRENT_MODEL: bool = False
    TOTAL_TRAIN_STEPS: int = int(3e5)
    SHOULD_LOG: bool = False if torch.cuda.is_available() else True

    TEST_SEED_OFFSET: int = 0

    CKPTS_TO_SAVE: int = 1

    # `LR` chosen by optimizing the performance of imitation learning
    LR: float = 0.0242


class BaseLightHouseExperimentConfig(ExperimentConfig, ABC):
    """Base experimental config."""

    _SENSOR_CACHE: Dict[Tuple[int, int, Optional[int], int], List[Sensor]] = {}

    def __init__(self, **kwargs):
        self.exp_params = LighthouseExperimentParams(**kwargs)

    def lr(self):
        return self.exp_params.LR

    def _action_space(self):
        return gym.spaces.Discrete(2 * self.exp_params.WORLD_DIM)

    def get_sensors(self):
        key = (
            self.exp_params.VIEW_RADIUS,
            self.exp_params.WORLD_DIM,
            (None if self.exp_params.RECURRENT_MODEL else self.exp_params.DEGREE),
            self.exp_params.EXPERT_VIEW_RADIUS,
        )

        assert (not self.exp_params.RECURRENT_MODEL) or self.exp_params.DEGREE == 1

        if key not in self._SENSOR_CACHE:
            sensors = [
                FactorialDesignCornerSensor(
                    view_radius=self.exp_params.VIEW_RADIUS,
                    world_dim=self.exp_params.WORLD_DIM,
                    degree=self.exp_params.DEGREE,
                )
            ]
            if self.exp_params.EXPERT_VIEW_RADIUS:
                sensors.append(
                    ExpertActionSensor(
                        expert_args={
                            "expert_view_radius": self.exp_params.EXPERT_VIEW_RADIUS,
                            "deterministic": True,
                        },
                        action_space=self._action_space(),
                    )
                )
            self._SENSOR_CACHE[key] = sensors

        return self._SENSOR_CACHE[key]

    def optimal_ave_ep_length(self):
        return LightHouseEnvironment.optimal_ave_ep_length(
            world_dim=self.exp_params.WORLD_DIM,
            world_radius=self.exp_params.WORLD_RADIUS,
            view_radius=self.exp_params.VIEW_RADIUS,
        )

    def get_early_stopping_criterion(self):
        optimal_ave_ep_length = self.optimal_ave_ep_length()

        return StopIfNearOptimal(
            optimal=optimal_ave_ep_length,
            deviation=optimal_ave_ep_length * 0.05,
            min_memory_size=50,
        )

    def rl_loss_default(self, alg: str, steps: Optional[int] = None):
        if alg == "ppo":
            assert steps is not None
            return {
                "loss": (PPO(clip_decay=LinearDecay(steps), **PPOConfig)),
                "num_mini_batch": 2,
                "update_repeats": 4,
            }
        elif alg == "a2c":
            return {
                "loss": A2C(**A2CConfig),
                "num_mini_batch": 1,
                "update_repeats": 1,
            }
        elif alg == "imitation":
            return {
                "loss": Imitation(),
                "num_mini_batch": 2,
                "update_repeats": 4,
            }
        else:
            raise NotImplementedError

    def _training_pipeline(
        self,
        named_losses: Dict[str, Union[Loss, Builder]],
        pipeline_stages: List[PipelineStage],
        num_mini_batch: int,
        update_repeats: int,
        lr: Optional[float] = None,
    ):
        # When using many mini-batches or update repeats, decrease the learning
        # rate so that the approximate size of the gradient update is similar.
        lr = self.exp_params.LR if lr is None else lr
        num_steps = 100
        metric_accumulate_interval = (
            self.exp_params.MAX_STEPS * 10
        )  # Log every 10 max length tasks
        gamma = 0.99

        if self.exp_params.CKPTS_TO_SAVE == 0:
            save_interval = None
        else:
            save_interval = math.ceil(
                sum(ps.max_stage_steps for ps in pipeline_stages)
                / self.exp_params.CKPTS_TO_SAVE
            )

        use_gae = "reinforce_loss" not in named_losses
        gae_lambda = 1.0
        max_grad_norm = 0.5

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=metric_accumulate_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses=named_losses,
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=None,
            should_log=self.exp_params.SHOULD_LOG,
            pipeline_stages=pipeline_stages,
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=self.exp_params.TOTAL_TRAIN_STEPS)}  # type: ignore
            ),
        )

    def machine_params(
        self, mode="train", gpu_id="default", n_train_processes="default", **kwargs
    ):
        if mode == "train":
            if n_train_processes == "default":
                nprocesses = self.exp_params.NUM_TRAIN_SAMPLERS
            else:
                nprocesses = n_train_processes
        elif mode == "valid":
            nprocesses = 0
        elif mode == "test":
            nprocesses = min(
                self.exp_params.NUM_TEST_TASKS, 500 if torch.cuda.is_available() else 50
            )
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        if gpu_id == "default":
            gpu_ids = [] if self.exp_params.GPU_ID is None else [self.exp_params.GPU_ID]
        else:
            gpu_ids = [gpu_id]

        return MachineParams(nprocesses=nprocesses, devices=gpu_ids)

    def create_model(self, **kwargs) -> nn.Module:
        sensors = self.get_sensors()
        if self.exp_params.RECURRENT_MODEL:
            return RNNActorCritic(
                input_uuid=sensors[0].uuid,
                action_space=self._action_space(),
                observation_space=SensorSuite(sensors).observation_spaces,
                rnn_type="LSTM",
            )
        else:
            return LinearActorCritic(
                input_uuid=sensors[0].uuid,
                action_space=self._action_space(),
                observation_space=SensorSuite(sensors).observation_spaces,
            )

    def make_sampler_fn(self, **kwargs) -> TaskSampler:
        return FindGoalLightHouseTaskSampler(**kwargs)

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return {
            "world_dim": self.exp_params.WORLD_DIM,
            "world_radius": self.exp_params.WORLD_RADIUS,
            "max_steps": self.exp_params.MAX_STEPS,
            "sensors": self.get_sensors(),
            "action_space": self._action_space(),
            "seed": seeds[process_ind] if seeds is not None else None,
        }

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        raise RuntimeError

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        max_tasks = self.exp_params.NUM_TEST_TASKS // total_processes + (
            process_ind < (self.exp_params.NUM_TEST_TASKS % total_processes)
        )
        task_seeds_list = [
            2 ** 31
            - 1
            + self.exp_params.TEST_SEED_OFFSET
            + process_ind
            + total_processes * i
            for i in range(max_tasks)
        ]

        assert min(task_seeds_list) >= 0 and max(task_seeds_list) <= 2 ** 32 - 1

        train_sampler_args = self.train_task_sampler_args(
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        return {
            **train_sampler_args,
            "task_seeds_list": task_seeds_list,
            "max_tasks": max_tasks,
            "deterministic_sampling": True,
        }
