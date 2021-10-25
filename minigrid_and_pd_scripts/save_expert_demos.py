import json
import os
import time
import typing

import babyai
import blosc
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from allenact.main import load_config, get_argument_parser
from allenact.utils.misc_utils import partition_sequence
from allenact.utils.system import get_logger
from allenact_plugins.minigrid_plugin.minigrid_tasks import MiniGridTaskSampler
from projects.advisor.minigrid_and_pd_experiments.base import BaseExperimentConfig

mp = mp.get_context("forkserver")
import queue
from setproctitle import setproctitle as ptitle
import numpy as np


def collect_demos(
    process_id: int, args, input_queue: mp.Queue, output_queue: mp.Queue,
):
    """Saves a collection of training demos."""
    ptitle("({}) Demo Saver".format(process_id))

    output_data_list = []
    try:
        cfg: BaseExperimentConfig

        config_kwargs = json.loads(
            "{}" if args.config_kwargs is None else args.config_kwargs
        )
        config_kwargs["MG_AGENT_VIEW_CHANNELS"] = 3
        config_kwargs["task_name"] = args.env_name
        args.config_kwargs = json.dumps(config_kwargs)

        cfg, _ = load_config(args)  # type: ignore

        wait_episodes = 100  # if torch.cuda.is_available() else 1

        task_sampler_args = cfg.train_task_sampler_args(
            process_ind=0, total_processes=0,
        )
        task_sampler = typing.cast(
            MiniGridTaskSampler,
            cfg.make_sampler_fn(
                **{
                    **task_sampler_args,
                    "task_seeds_list": ["UNDEFINED"],
                    "deterministic_sampling": True,
                    "repeat_failed_task_for_min_steps": 0,
                }
            ),
        )

        while True:
            seeds = input_queue.get(timeout=1)

            for seed in seeds:
                task_sampler.task_seeds_list[0] = seed

                task = task_sampler.next_task()
                images = []
                actions = []
                directions = []

                def append_values():
                    assert not task.is_done()

                    obs = task.get_observations()
                    images.append(obs["minigrid_ego_image"])
                    actions.append(int(obs["expert_action"].reshape(-1)[0]))
                    directions.append(task.env.agent_dir)

                while not task.is_done():
                    append_values()
                    task.step(action=actions[-1])

                output_data_list.append(
                    {
                        "seed": seed,
                        "images": blosc.pack_array(np.array(images)),
                        "actions": actions,
                        "directions": directions,
                    }
                )

                if len(output_data_list) >= wait_episodes:
                    output_queue.put(output_data_list)
                    # print(
                    #     sum(len(od["actions"]) for od in output_data_list)
                    #     / len(output_data_list)
                    # )
                    output_data_list = []
    except queue.Empty:
        if len(output_data_list) != 0:
            output_queue.put(output_data_list)

        get_logger().info("Queue empty for worker {}, exiting.".format(process_id))


def create_demos(args, nprocesses: int, min_demos: int):
    assert args.experiment in ["", "bc"], "`--experiment` must be either empty or 'bc'."
    assert os.path.relpath(args.output_dir) != ""

    task_name = args.env_name

    ptitle("Master (DEMOs {})".format(" and ".join(task_name)))

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    demos_save_path = os.path.join(output_dir, "MiniGrid-{}-v0.pkl".format(task_name))

    if os.path.exists(demos_save_path):
        demos_list = babyai.utils.load_demos(demos_save_path)
        if len(demos_list) > min_demos:
            min_demos = len(demos_list)
        demos_list.extend([None] * (min_demos - len(demos_list)))
        remaining_seeds = set(i for i, d in enumerate(demos_list) if d is None)
    else:
        demos_list = [None] * min_demos
        remaining_seeds = set(range(min_demos))

    if len(remaining_seeds) == 0:
        print(f"No more demos to save for task {task_name}")
        return len(demos_list), sum([len(dl[3]) for dl in demos_list])

    print(f"Beginning to save demos with {len(remaining_seeds)} remaining")

    input_queue = mp.Queue()
    for seeds in partition_sequence(
        list(remaining_seeds), min(2 ** 15 - 1, len(remaining_seeds))
    ):
        # Annoyingly a mp.Queue can hold a max of 2**15 - 1 items so we have to do this hack
        input_queue.put(seeds)

    output_queue = mp.Queue()

    processes = []
    for i in range(min(nprocesses, len(remaining_seeds))):
        processes.append(
            mp.Process(
                target=collect_demos,
                kwargs=dict(
                    process_id=i,
                    args=args,
                    input_queue=input_queue,
                    output_queue=output_queue,
                ),
            )
        )
        processes[-1].start()
        time.sleep(0.1)

    with tqdm(total=len(remaining_seeds)) as pbar:
        total_demos_created = sum(d is not None for d in demos_list)
        while len(remaining_seeds) != 0:
            try:
                run_data_list = output_queue.get(timeout=60)
                for run_data in run_data_list:
                    remaining_seeds.remove(run_data["seed"])

                    demos_list[run_data["seed"]] = (
                        "",
                        run_data["images"],
                        run_data["directions"],
                        run_data["actions"],
                    )

                    total_demos_created += 1
                    if total_demos_created % 10000 == 0:
                        babyai.utils.save_demos(demos_list, demos_save_path)

                    pbar.update(1)
            except queue.Empty as _:
                print("No demo saved for 60 seconds")

    babyai.utils.save_demos(demos_list, demos_save_path)

    for p in processes:
        try:
            p.join(1)
        except Exception as _:
            pass

    print("Single stage of saving data is done!")

    return len(demos_list), sum([len(dl[3]) for dl in demos_list])


if __name__ == "__main__":
    """Run this with the following command (from the package's root):

    Command:
    python minigrid_and_pd_scripts/save_expert_demos.py bc \
    -b minigrid_and_pd_experiments/ \
    -o minigrid_data/minigrid_demos \
    --env_name CrossingS25N10

    Generate all the commands:
    ```python
    ns = [
        "CrossingS25N10",
        "WallCrossingS25N10",
        "WallCrossingCorruptExpertS25N10",
        "LavaCrossingCorruptExpertS15N7",
        "AskForHelpSimpleCrossing",
        "AskForHelpSimpleCrossingOnce",
        "AskForHelpLavaCrossingOnce",
        "AskForHelpLavaCrossingSmall",
    ]
    s = "python minigrid_and_pd_scripts/save_expert_demos.py bc -b minigrid_and_pd_experiments/ -o minigrid_data/minigrid_demos --env_name {}"
    cmd = " ; ".join([s.format(n) for n in ns])
    print(cmd)
    ```
    """
    parser = get_argument_parser()
    parser.add_argument(
        "--env_name", type=str, required=True,
    )
    args = parser.parse_args()

    initial_processes = min(6 if not torch.cuda.is_available() else 10, mp.cpu_count())
    nprocesses = min(6 if not torch.cuda.is_available() else 56, mp.cpu_count())

    min_demos = int(20)
    count = 0
    while count < int(1e6):
        min_demos, count = create_demos(
            args,
            nprocesses=initial_processes if count == 0 else nprocesses,
            min_demos=min_demos,
        )
        print(f"{count} frames saved so far.")
        min_demos = max(int(1e6 / (count / min_demos)), min_demos) + 100

    print("Saving explore combination data is done!")
