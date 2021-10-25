import json
import os
from typing import cast, Dict

import tqdm

from advisor_constants import ADVISOR_TOP_LEVEL_DIR
from allenact.main import get_argument_parser, load_config
from allenact.utils.experiment_utils import set_seed, ScalarMeanTracker
from minigrid_and_pd_experiments.base import BaseExperimentConfig

TASK_TO_RANDOM_PERFORMANCE = {
    "PoisonedDoors": {
        "ep_length": 2.091,
        "reward": -0.464,
        "chose_door_0": 0.254,
        "chose_door_1": 0.242,
        "chose_door_2": 0.25,
        "chose_door_3": 0.254,
        "chose_no_door": 0.0,
        "chose_good_door": 0.257,
        "opened_lock": 0.0,
        "success": 0.257,
        "max_comb_correct": 0.2125984251968504,
    },
    "CrossingS25N10": {"ep_length": 25.908, "reward": 0.0, "success": 0.0},
    "WallCrossingS25N10": {"ep_length": 2466.98, "reward": 0.0168872, "success": 0.05},
    "WallCrossingCorruptExpertS25N10": {
        "ep_length": 2463.183,
        "reward": 0.018654119999999996,
        "success": 0.054,
    },
    "LavaCrossingCorruptExpertS15N7": {
        "ep_length": 19.317,
        "reward": 0.0,
        "success": 0.0,
    },
    "AskForHelpSimpleCrossing": {
        "ep_length": 882.099,
        "reward": 0.024601,
        "explored_count": 20.073,
        "final_distance": 11.309,
        "success": 0.067,
        "toggle_percent": 0.2505814965872374,
    },
    "AskForHelpSimpleCrossingOnce": {
        "ep_length": 2484.158,
        "reward": 0.008303119999999999,
        "explored_count": 45.412,
        "final_distance": 22.958,
        "success": 0.026,
        "toggle_percent": 0.2500501483796506,
    },
    "AskForHelpLavaCrossingOnce": {
        "ep_length": 26.422,
        "reward": 0.0,
        "explored_count": 3.952,
        "final_distance": 21.539,
        "success": 0.0,
        "toggle_percent": 0.2231268780071966,
    },
    "AskForHelpLavaCrossingSmall": {
        "ep_length": 19.678,
        "reward": 0.0,
        "explored_count": 3.345,
        "final_distance": 9.904,
        "success": 0.0,
        "toggle_percent": 0.20499024899878812,
    },
}

_TASK_NAMES = [
    "PoisonedDoors",
    "CrossingS25N10",
    "WallCrossingS25N10",
    "WallCrossingCorruptExpertS25N10",
    "LavaCrossingCorruptExpertS15N7",
    "AskForHelpSimpleCrossing",
    "AskForHelpSimpleCrossingOnce",
    "AskForHelpLavaCrossingOnce",
    "AskForHelpLavaCrossingSmall",
]


if __name__ == "__main__":

    for task_name in _TASK_NAMES:

        config_kwargs = {"task_name": task_name}

        exp_path = os.path.join(
            ADVISOR_TOP_LEVEL_DIR, "minigrid_and_pd_experiments/bc.py"
        )
        args_list = [
            exp_path,
            "--config_kwargs",
            json.dumps(config_kwargs),
        ]
        parser = get_argument_parser()
        args = parser.parse_args(args=args_list)

        cfg: BaseExperimentConfig = cast(BaseExperimentConfig, load_config(args)[0])

        test_sampler_kwargs = cfg.test_task_sampler_args(
            process_ind=0, total_processes=1, seeds=[0]
        )

        task_sampler = cfg.make_sampler_fn(**test_sampler_kwargs)

        metrics_list = []
        means_tracker = ScalarMeanTracker()
        k = 0
        print(f"Starting random performance test for {task_name}")
        pbar = tqdm.tqdm(total=cfg.exp_params.NUM_TEST_TASKS)
        while True:
            set_seed(k)
            k += 1
            task = task_sampler.next_task()
            if task is None:
                break

            while not task.is_done():
                task.step(action=task.action_space.sample())

            metrics_list.append(task.metrics())
            means_tracker.add_scalars(
                {k: v for k, v in metrics_list[-1].items() if not isinstance(v, Dict)}
            )
            pbar.update(1)

        pbar.close()
        print()

        print(f"Random performance for {task_name}:")
        print(dict(means_tracker.means()))
        print("\n")
