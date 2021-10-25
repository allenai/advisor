import abc
import math
import os
from typing import (
    Optional,
    List,
    Any,
    Dict,
    cast,
    Sequence,
    Callable,
    Union,
    NamedTuple,
)

import gym
import torch
from gym_minigrid.minigrid import Lava, WorldObj, Wall
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses import PPO, A2C
from allenact.algorithms.onpolicy_sync.losses.a2cacktr import A2CConfig
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.misc import Loss
from allenact.base_abstractions.sensor import SensorSuite, Sensor, ExpertActionSensor
from allenact.embodiedai.models.basic_models import LinearActorCritic
from allenact.utils.experiment_utils import (
    LinearDecay,
    Builder,
    PipelineStage,
    TrainingPipeline,
)
from allenact_plugins.lighthouse_plugin.lighthouse_models import (
    LinearAdvisorActorCritic,
)
from allenact_plugins.minigrid_plugin.minigrid_environments import (
    FastCrossing,
    AskForHelpSimpleCrossing,
)
from allenact_plugins.minigrid_plugin.minigrid_models import MiniGridSimpleConvRNN
from allenact_plugins.minigrid_plugin.minigrid_offpolicy import (
    create_minigrid_offpolicy_data_iterator,
)
from allenact_plugins.minigrid_plugin.minigrid_sensors import EgocentricMiniGridSensor
from allenact_plugins.minigrid_plugin.minigrid_tasks import (
    MiniGridTaskSampler,
    MiniGridTask,
    AskForHelpSimpleCrossingTask,
)
from poisoneddoors_plugin.poisoneddoors_models import RNNActorCriticWithEmbed
from poisoneddoors_plugin.poisoneddoors_offpolicy import (
    create_poisoneddoors_offpolicy_data_iterator,
)
from poisoneddoors_plugin.poisoneddoors_sensors import PoisonedDoorCurrentStateSensor
from poisoneddoors_plugin.poisoneddoors_tasks import (
    PoisonedDoorsEnvironment,
    PoisonedDoorsTask,
    PoisonedDoorsTaskSampler,
)
from projects.advisor.minigrid_constants import MINIGRID_EXPERT_TRAJECTORIES_DIR


class MiniGridAndPDExperimentParams(NamedTuple):
    TASK_NAME: str

    # Default MiniGrid values
    MG_AGENT_VIEW_SIZE: int = 7
    MG_AGENT_VIEW_CHANNELS: int = 3

    # Default Poisoned Doors values
    PD_MAX_STEPS: int = 100

    # Training params
    NUM_TRAIN_SAMPLERS: int = 20  # if torch.cuda.is_available() else 1
    ROLLOUT_STEPS: int = 100
    MG_TOTAL_TRAIN_STEPS = int(1e6)
    PD_TOTAL_TRAIN_STEPS = int(3e5)
    NUM_TRAIN_TASKS: int = None
    NUM_TEST_TASKS: int = 1000
    GPU_ID: Optional[int] = 1 if torch.cuda.is_available() else None
    USE_EXPERT: bool = False
    RNN_TYPE: str = "LSTM"
    CACHE_GRAPHS: bool = False
    SHOULD_LOG = True
    TEST_SEED_OFFSET: int = 0

    # Hyperparameters
    LR: Optional[float] = None
    TF_RATIO: Optional[float] = None
    FIXED_ALPHA: Optional[float] = None
    ALPHA_START: Optional[float] = None
    ALPHA_STOP: Optional[float] = None

    # Auxiliary head parameters
    INCLUDE_AUXILIARY_HEAD: bool = False
    SAME_INIT_VALS_FOR_ADVISOR_HEAD: bool = False

    # Logging / saving
    METRIC_ACCUMULATE_INTERVAL = 10000 if torch.cuda.is_available() else 1000
    CKPTS_TO_SAVE = 4


class BaseExperimentConfig(ExperimentConfig):
    """Base experiment."""

    def __init__(self, task_name: str, **kwargs):
        self.exp_params = MiniGridAndPDExperimentParams(TASK_NAME=task_name, **kwargs)

    @property
    def task_name(self):
        return self.exp_params.TASK_NAME

    def total_train_steps(self) -> int:
        task_info = self.task_info()
        return task_info["total_train_steps"]

    def task_info(self):
        """All information needed about the underlying task.

        # Returns

         Dictionary of useful information:
            - env_info: used to initialize the environment
            - tag: string to use for logging
            - env_class: callable of the underlying mini-grid / poisoned doors environment class
            - task_class: callable of the corresponding task class
        """
        name = self.task_name
        output_data = dict()

        if name == "PoisonedDoors":
            # Specific base parameters
            num_doors = 4
            combination_length = 10
            extra_tag = self.extra_tag()
            # Parameters needed for other functions
            output_data["env_info"] = {
                "num_doors": num_doors,
                "combination_length": combination_length,
            }
            output_data["task_sampler_args"] = {
                **output_data["env_info"],
                "max_steps": self.exp_params.PD_MAX_STEPS,
            }
            output_data["tag"] = "PoisonedDoorsN{}{}".format(num_doors, extra_tag,)
            output_data["env_class"] = PoisonedDoorsEnvironment
            output_data["task_class"] = PoisonedDoorsTask
            output_data["task_sampler_class"] = PoisonedDoorsTaskSampler
        elif name == "CrossingS25N10":
            # Specific base parameters
            grid_size = 25
            num_crossings = 10
            obstacle_type: Callable[[], WorldObj] = Lava
            # Parameters needed for other functions
            output_data["env_info"] = {
                "size": grid_size,
                "num_crossings": num_crossings,
                "obstacle_type": obstacle_type,
            }
            output_data["tag"] = "Crossing{}S{}N{}{}".format(
                obstacle_type().__class__.__name__,
                grid_size,
                num_crossings,
                self.extra_tag(),
            )
            output_data["task_sampler_args"] = {
                "repeat_failed_task_for_min_steps": 1000
            }
            output_data["env_class"] = FastCrossing
            output_data["task_class"] = MiniGridTask
            output_data["task_sampler_class"] = MiniGridTaskSampler

        elif name == "WallCrossingS25N10":
            # Specific base parameters
            grid_size = 25
            num_crossings = 10
            obstacle_type: Callable[[], WorldObj] = Wall
            # Parameters needed for other functions
            output_data["env_info"] = {
                "size": grid_size,
                "num_crossings": num_crossings,
                "obstacle_type": obstacle_type,
            }
            output_data["tag"] = "Crossing{}S{}N{}{}".format(
                obstacle_type().__class__.__name__,
                grid_size,
                num_crossings,
                self.extra_tag(),
            )
            # # Each episode takes 4 * 25 * 25 = 2500 steps already, so no need to set
            # # repeat_failed_task_for_min_steps
            # output_data["task_sampler_args"] = {
            #     "repeat_failed_task_for_min_steps": 1000
            # }
            output_data["env_class"] = FastCrossing
            output_data["task_class"] = MiniGridTask
            output_data["task_sampler_class"] = MiniGridTaskSampler

        elif name == "WallCrossingCorruptExpertS25N10":
            # Specific base parameters
            grid_size = 25
            num_crossings = 10
            corrupt_expert_within_actions_of_goal = 15

            obstacle_type: Callable[[], WorldObj] = Wall
            # Parameters needed for other functions
            output_data["env_info"] = {
                "size": grid_size,
                "num_crossings": num_crossings,
                "obstacle_type": obstacle_type,
            }
            output_data["tag"] = "WallCrossingCorruptExpert{}S{}N{}C{}{}".format(
                obstacle_type().__class__.__name__,
                grid_size,
                num_crossings,
                corrupt_expert_within_actions_of_goal,
                self.extra_tag(),
            )
            # # Each episode takes 4 * 25 * 25 = 2500 steps already, so no need to set
            # # repeat_failed_task_for_min_steps
            output_data["task_sampler_args"] = {
                "extra_task_kwargs": {
                    "corrupt_expert_within_actions_of_goal": corrupt_expert_within_actions_of_goal
                }
            }
            # output_data["task_sampler_args"] = {
            #     "repeat_failed_task_for_min_steps": 1000
            # }
            output_data["env_class"] = FastCrossing
            output_data["task_class"] = MiniGridTask
            output_data["task_sampler_class"] = MiniGridTaskSampler

        elif name == "LavaCrossingCorruptExpertS15N7":
            # Specific base parameters
            grid_size = 15
            num_crossings = 7
            corrupt_expert_within_actions_of_goal = 10
            obstacle_type: Callable[[], WorldObj] = Lava

            # Parameters needed for other functions
            output_data["env_info"] = {
                "size": grid_size,
                "num_crossings": num_crossings,
                "obstacle_type": obstacle_type,
            }
            output_data["tag"] = "LavaCrossingCorruptExpert{}S{}N{}C{}{}".format(
                obstacle_type().__class__.__name__,
                grid_size,
                num_crossings,
                corrupt_expert_within_actions_of_goal,
                self.extra_tag(),
            )
            # # Each episode takes 4 * 25 * 25 = 2500 steps already, so no need to set
            # # repeat_failed_task_for_min_steps
            output_data["task_sampler_args"] = {
                "extra_task_kwargs": {
                    "corrupt_expert_within_actions_of_goal": corrupt_expert_within_actions_of_goal
                },
                "repeat_failed_task_for_min_steps": 1000,
            }
            output_data["env_class"] = FastCrossing
            output_data["task_class"] = MiniGridTask
            output_data["task_sampler_class"] = MiniGridTaskSampler

        elif name == "AskForHelpSimpleCrossing":
            # Specific base parameters
            grid_size = 15
            num_crossings = 7
            obstacle_type: Callable[[], WorldObj] = Wall
            # Parameters needed for other functions
            output_data["env_info"] = {
                "size": grid_size,
                "num_crossings": num_crossings,
                "obstacle_type": obstacle_type,
            }
            output_data["tag"] = "AskForHelpSimpleCrossing{}S{}N{}{}".format(
                obstacle_type().__class__.__name__,
                grid_size,
                num_crossings,
                self.extra_tag(),
            )
            # output_data["task_sampler_args"] = {
            #     "repeat_failed_task_for_min_steps": 1000
            # }
            output_data["env_class"] = AskForHelpSimpleCrossing
            output_data["task_class"] = AskForHelpSimpleCrossingTask
            output_data["task_sampler_class"] = MiniGridTaskSampler

        elif name == "AskForHelpSimpleCrossingOnce":
            # Specific base parameters
            grid_size = 25
            num_crossings = 10
            toggle_is_permanent = True
            obstacle_type: Callable[[], WorldObj] = Wall
            # Parameters needed for other functions
            output_data["env_info"] = {
                "size": grid_size,
                "num_crossings": num_crossings,
                "obstacle_type": obstacle_type,
                "toggle_is_permenant": toggle_is_permanent,
            }
            output_data["tag"] = "AskForHelpSimpleCrossingOnce{}S{}N{}{}".format(
                obstacle_type().__class__.__name__,
                grid_size,
                num_crossings,
                self.extra_tag(),
            )
            output_data["task_sampler_args"] = {
                "repeat_failed_task_for_min_steps": 1000
            }
            output_data["env_class"] = AskForHelpSimpleCrossing
            output_data["task_class"] = AskForHelpSimpleCrossingTask
            output_data["task_sampler_class"] = MiniGridTaskSampler

        elif name == "AskForHelpLavaCrossingOnce":
            # Specific base parameters
            grid_size = 15
            num_crossings = 7
            toggle_is_permanent = True
            obstacle_type: Callable[[], WorldObj] = Lava
            # Parameters needed for other functions
            output_data["env_info"] = {
                "size": grid_size,
                "num_crossings": num_crossings,
                "obstacle_type": obstacle_type,
                "toggle_is_permenant": toggle_is_permanent,
            }
            output_data["tag"] = "AskForHelpLavaCrossingOnce{}S{}N{}{}".format(
                obstacle_type().__class__.__name__,
                grid_size,
                num_crossings,
                self.extra_tag(),
            )
            output_data["task_sampler_args"] = {
                "repeat_failed_task_for_min_steps": 1000
            }
            output_data["env_class"] = AskForHelpSimpleCrossing
            output_data["task_class"] = AskForHelpSimpleCrossingTask
            output_data["task_sampler_class"] = MiniGridTaskSampler

        elif name == "AskForHelpLavaCrossingSmall":
            # Specific base parameters
            grid_size = 9
            num_crossings = 4
            obstacle_type: Callable[[], WorldObj] = Lava
            # Parameters needed for other functions
            output_data["env_info"] = {
                "size": grid_size,
                "num_crossings": num_crossings,
                "obstacle_type": obstacle_type,
            }
            output_data["tag"] = "AskForHelpLavaCrossingSmall{}S{}N{}{}".format(
                obstacle_type().__class__.__name__,
                grid_size,
                num_crossings,
                self.extra_tag(),
            )
            output_data["task_sampler_args"] = {
                "repeat_failed_task_for_min_steps": 1000
            }
            output_data["env_class"] = AskForHelpSimpleCrossing
            output_data["task_class"] = AskForHelpSimpleCrossingTask
            output_data["task_sampler_class"] = MiniGridTaskSampler

        else:
            raise NotImplementedError("Haven't implemented {}".format(name))

        if name == "PoisonedDoors":
            output_data["total_train_steps"] = self.exp_params.PD_TOTAL_TRAIN_STEPS
        else:
            # MiniGrid total train steps
            output_data["total_train_steps"] = self.exp_params.MG_TOTAL_TRAIN_STEPS

        output_data["name"] = name
        return output_data

    def tag(self):
        return self.task_info()["tag"]

    @abc.abstractmethod
    def extra_tag(self):
        raise NotImplementedError

    def get_sensors(self) -> Sequence[Sensor]:
        task_info = self.task_info()

        if task_info["name"] == "PoisonedDoors":
            action_space = gym.spaces.Discrete(
                len(
                    task_info["task_class"].class_action_names(
                        num_doors=task_info["env_info"]["num_doors"]
                    )
                )
            )
            return [PoisonedDoorCurrentStateSensor()] + (
                [ExpertActionSensor(action_space=action_space)]
                if self.exp_params.USE_EXPERT
                else []
            )
        else:
            # Sensors for MiniGrid tasks
            action_space = gym.spaces.Discrete(
                len(task_info["task_class"].class_action_names())
            )
            return [
                EgocentricMiniGridSensor(
                    agent_view_size=self.exp_params.MG_AGENT_VIEW_SIZE,
                    view_channels=self.exp_params.MG_AGENT_VIEW_CHANNELS,
                )
            ] + (
                [ExpertActionSensor(action_space=action_space)]
                if self.exp_params.USE_EXPERT
                else []
            )

    def machine_params(self, mode="train", gpu_id="default", **kwargs):
        if mode == "train":
            nprocesses = self.exp_params.NUM_TRAIN_SAMPLERS
        elif mode == "valid":
            nprocesses = 0
        elif mode == "test":
            nprocesses = min(
                self.exp_params.NUM_TEST_TASKS, 500 if torch.cuda.is_available() else 50
            )
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        gpu_ids = [] if self.exp_params.GPU_ID is None else [self.exp_params.GPU_ID]

        return MachineParams(nprocesses=nprocesses, devices=gpu_ids)

    def create_model(self, **kwargs) -> nn.Module:
        sensors = self.get_sensors()
        task_info = self.task_info()
        if task_info["name"] == "PoisonedDoors":
            return RNNActorCriticWithEmbed(
                input_uuid=sensors[0].uuid,
                num_embeddings=4,
                embedding_dim=128,
                input_len=1,
                action_space=gym.spaces.Discrete(
                    3 + task_info["env_info"]["num_doors"]
                ),
                observation_space=SensorSuite(sensors).observation_spaces,
                rnn_type=self.exp_params.RNN_TYPE,
                head_type=LinearActorCritic
                if not self.exp_params.INCLUDE_AUXILIARY_HEAD
                else Builder(  # type: ignore
                    LinearAdvisorActorCritic,
                    kwargs={
                        "ensure_same_init_aux_weights": self.exp_params.SAME_INIT_VALS_FOR_ADVISOR_HEAD
                    },
                ),
            )
        else:
            # Model for MiniGrid tasks
            return MiniGridSimpleConvRNN(
                action_space=gym.spaces.Discrete(
                    len(task_info["task_class"].class_action_names())
                ),
                num_objects=cast(EgocentricMiniGridSensor, sensors[0]).num_objects,
                num_colors=cast(EgocentricMiniGridSensor, sensors[0]).num_colors,
                num_states=cast(EgocentricMiniGridSensor, sensors[0]).num_states,
                observation_space=SensorSuite(sensors).observation_spaces,
                hidden_size=128,
                rnn_type=self.exp_params.RNN_TYPE,
                head_type=LinearActorCritic
                if not self.exp_params.INCLUDE_AUXILIARY_HEAD
                else Builder(  # type: ignore
                    LinearAdvisorActorCritic,
                    kwargs={
                        "ensure_same_init_aux_weights": self.exp_params.SAME_INIT_VALS_FOR_ADVISOR_HEAD
                    },
                ),
            )

    def make_sampler_fn(
        self, **kwargs
    ) -> Union[PoisonedDoorsTaskSampler, MiniGridTaskSampler]:
        return self.task_info()["task_sampler_class"](**kwargs)

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        info = self.task_info()
        if info["name"] == "PoisonedDoors":
            args_dict = {
                "sensors": self.get_sensors(),
                "env_class": info.get("env_class"),
                "env_info": info.get("env_info"),
                "task_class": info["task_class"],
            }
        else:
            args_dict = {
                "sensors": self.get_sensors(),
                "env_class": info.get("env_class"),
                "env_info": info.get("env_info"),
                "cache_graphs": self.exp_params.CACHE_GRAPHS,
                "task_class": info["task_class"],
            }
        if "task_sampler_args" in info:
            args_dict.update(info["task_sampler_args"])

        if self.exp_params.NUM_TRAIN_TASKS:
            args_dict["max_tasks"] = self.exp_params.NUM_TRAIN_TASKS
        return args_dict

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
        if "repeat_failed_task_for_min_steps" in train_sampler_args:
            del train_sampler_args["repeat_failed_task_for_min_steps"]
        return {
            **train_sampler_args,
            "task_seeds_list": task_seeds_list,
            "max_tasks": max_tasks,
            "deterministic_sampling": True,
            "sensors": [
                s for s in train_sampler_args["sensors"] if "Expert" not in str(type(s))
            ],
        }

    def offpolicy_demo_defaults(self, also_using_ppo: bool):
        ppo_defaults = self.rl_loss_default("ppo", 1)
        assert ppo_defaults["update_repeats"] % 2 == 0
        output_data = {}
        task_info = self.task_info()
        if task_info["name"] == "PoisonedDoors":
            output_data.update(
                {
                    "data_iterator_builder": lambda: create_poisoneddoors_offpolicy_data_iterator(
                        num_doors=task_info["env_info"]["num_doors"],
                        nrollouts=self.exp_params.NUM_TRAIN_SAMPLERS
                        // ppo_defaults["num_mini_batch"],
                        rollout_len=self.exp_params.ROLLOUT_STEPS,
                        dataset_size=task_info["total_train_steps"],
                    ),
                }
            )
        else:
            # Off-policy defaults for MiniGrid tasks
            output_data.update(
                {
                    "data_iterator_builder": lambda: create_minigrid_offpolicy_data_iterator(
                        path=os.path.join(
                            MINIGRID_EXPERT_TRAJECTORIES_DIR,
                            "MiniGrid-{}-v0{}.pkl".format(task_info["name"], "",),
                        ),
                        nrollouts=self.exp_params.NUM_TRAIN_SAMPLERS
                        // ppo_defaults["num_mini_batch"],
                        rollout_len=self.exp_params.ROLLOUT_STEPS,
                        instr_len=None,
                        restrict_max_steps_in_dataset=task_info["total_train_steps"],
                    ),
                }
            )
        # Off-policy defaults common to Poisoned Doors and MiniGrid tasks
        output_data.update(
            {
                "ppo_update_repeats": ppo_defaults["update_repeats"] // 2
                if also_using_ppo
                else 0,
                "ppo_num_mini_batch": ppo_defaults["num_mini_batch"]
                if also_using_ppo
                else 0,
                "offpolicy_updates": ppo_defaults["num_mini_batch"]
                * (
                    ppo_defaults["update_repeats"] // 2
                    if also_using_ppo
                    else ppo_defaults["update_repeats"]
                ),
            }
        )
        return output_data

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
                "num_mini_batch": 2,  # if torch.cuda.is_available() else 1,
                "update_repeats": 4,
            }
        else:
            raise NotImplementedError

    def _training_pipeline(
        self,
        named_losses: Dict[str, Union[Loss, Builder]],
        pipeline_stages: List[PipelineStage],
        num_mini_batch: int,
        update_repeats: Optional[int],
    ):
        # When using many mini-batches or update repeats, decrease the learning
        # rate so that the approximate size of the gradient update is similar.
        lr = self.exp_params.LR
        num_steps = self.exp_params.ROLLOUT_STEPS
        metric_accumulate_interval = self.exp_params.METRIC_ACCUMULATE_INTERVAL

        gamma = 0.99

        use_gae = "reinforce_loss" not in named_losses
        gae_lambda = 1.0
        max_grad_norm = 0.5

        total_train_steps = self.task_info()["total_train_steps"]

        if self.exp_params.CKPTS_TO_SAVE == 0:
            save_interval = None
        else:
            save_interval = math.ceil(total_train_steps / self.exp_params.CKPTS_TO_SAVE)

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
                LambdaLR, {"lr_lambda": LinearDecay(steps=total_train_steps)}  # type: ignore
            ),
        )
