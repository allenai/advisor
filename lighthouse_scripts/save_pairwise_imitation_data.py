import glob
import json
import math
import os
import shutil
import time
from typing import Optional

import torch
import torch.multiprocessing as mp

from allenact.algorithms.onpolicy_sync.runner import OnPolicyRunner
from allenact.main import get_args, init_logging, load_config
from allenact_plugins.lighthouse_plugin.lighthouse_environment import (
    LightHouseEnvironment,
)
from projects.advisor.lighthouse_experiments.base import (
    BaseLightHouseExperimentConfig,
    LighthouseExperimentParams,
)

mp = mp.get_context("forkserver")
import queue
from setproctitle import setproctitle as ptitle
import pandas as pd
import numpy as np
from allenact.utils.system import get_logger, update_log_level


def iteratively_run_lighthouse_experiments(
    process_id: int,
    gpu_id: Optional[int],
    args,
    world_dim: int,
    world_radius: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    log_level: str,
    test_seed_offset: Optional[int] = None,
):
    """Iteratively train and test lighthouse models with different levels of
    supervision.

    This function is meant to be run as a subprocess. It iteratively samples
    `input_queue` from a queue that define the next experiment to run (e.g. the agent's view
    radius, the expert's view radius, and any seed). It then runs this experiment
    and adds the results of the experiment to the `output_queue` which is collated
    by the main process.

    # Attributes

    process_id : This process' id.
    gpu_id : The gpu to run experiments on.
    args : Command line arguments specifying the experiment config to run. E.g.
        `projects.advisor.lighthouse_experiments.advisor`. Details of this
        experiment config, for instance the agent's `view_radius` config are modified
        by this function based on the values from the `input_queue`.
    world_dim : The world dimension used in all experiments.
    world_radius : The world radius used in all experiments.
    input_queue : The queue from which experiment details are taken.
    output_queue : The queue into which the results of running an experiment are saved.

    test_seed_offset : If not `None`, used to redefine the `TEST_SEED_OFFSET` class constant
        associated with the experiment config.
    """
    ptitle("({}) Create Im. Mat. Runner".format(process_id))

    init_logging(log_level)

    def log_info(msg):
        update_log_level(logger=get_logger(), human_log_level="info")
        get_logger().info(msg)
        update_log_level(logger=get_logger(), human_log_level=log_level)

    try:
        while True:
            # Sample a new set of values defining the new experiment to run
            view_radius, expert_view_radius, seed, lr = input_queue.get(timeout=1)
            optimal_ave_ep_length = LightHouseEnvironment.optimal_ave_ep_length(
                world_dim=world_dim, world_radius=world_radius, view_radius=view_radius
            )

            args.config_kwargs = json.dumps(
                {
                    "GPU_ID": gpu_id,
                    "TEST_SEED_OFFSET": test_seed_offset
                    if test_seed_offset is not None
                    else 0,
                    "LR": lr if lr is not None else LighthouseExperimentParams().LR,
                    "VIEW_RADIUS": view_radius,
                    "EXPERT_VIEW_RADIUS": expert_view_radius,
                }
            )

            # Grab the experiment config and set its GPU_ID.
            cfg: BaseLightHouseExperimentConfig
            cfg, _ = load_config(args)  # type: ignore

            log_info(
                f"Running with (view, expert view, seed, lr) ="
                f" ({view_radius}, {expert_view_radius}, {seed}, {cfg.exp_params.LR:.3g})."
                f" Target optimal ep length: {optimal_ave_ep_length}."
            )
            assert args.seed is None

            # Train the agent based on the experiment config.
            runner = OnPolicyRunner(
                config=cfg,
                output_dir=args.output_dir,
                loaded_config_src_files=None,
                seed=seed,
                mode="train",
                mp_ctx=mp,
                disable_tensorboard=True,
                disable_config_saving=True,
            )

            train_start_time_str = runner.start_train(
                max_sampler_processes_per_worker=1,
                save_ckpt_after_every_pipeline_stage=False,
            )

            ckpt_dir = runner.checkpoint_dir(
                start_time_str=train_start_time_str, create_if_none=False
            )

            log_info(
                f"Running testing with (view, expert view, seed, lr) ="
                f" ({view_radius}, {expert_view_radius}, {seed}, {cfg.exp_params.LR:.3g})."
            )

            runner.mode = "test"
            test_results = runner.start_test(
                checkpoint_path_dir_or_pattern=ckpt_dir,
                max_sampler_processes_per_worker=1,
            )
            runner.close()

            # Remove the checkpoint file saved above as we no longer need it.
            assert len(glob.glob(os.path.join(ckpt_dir, "*"))) == len(
                glob.glob(os.path.join(ckpt_dir, "*.pt"))
            )
            shutil.rmtree(ckpt_dir)

            log_info(
                f"Testing complete for (view, expert view, seed, lr) ="
                f" ({view_radius}, {expert_view_radius}, {seed}, {cfg.exp_params.LR:.3g})."
            )

            # Put results from test evaluation into the output queue to be
            # collated by the main thread.
            output_queue.put(
                (
                    (view_radius, expert_view_radius, seed, lr),
                    {
                        "view_radius": int(view_radius),
                        "expert_view_radius": None
                        if expert_view_radius is None
                        else int(expert_view_radius),
                        "optimal": optimal_ave_ep_length,
                        "reached_near_optimal": 1
                        * (test_results[0]["ep_length"] < optimal_ave_ep_length * 1.1),
                        "avg_ep_length": float(test_results[0]["ep_length"]),
                        "train_steps": int(test_results[0]["training_steps"]),
                        "seed": seed,
                        "start_time_str": train_start_time_str,
                        "lr": lr,
                    },
                )
            )
    except queue.Empty:
        log_info("Queue empty for worker {}, exiting.".format(process_id))


if __name__ == "__main__":
    """Controls the master process that: (1) Instantiates several subprocesses
    which run the experiments. (2) Collates the results from the experiments
    run in the subprocesses.

    Get command line arguments that define the experiment. For instance, we might run
    this script (from within the `advisor` directory), with arguments

    ```
    python projects/advisor/lighthouse_scripts/save_pairwise_imitation_data.py \
    projects/advisor/lighthouse_experiments/dagger_then_ppo.py \
    -m 1 \
    --output_dir hp_runs/lighthouse \
    --log_level error
    ```

    And this will exhaustively train using the `dagger_then_ppo` experiment
    with various agent/expert view radii.

    Generate all commands as follows:
    ```python
    import glob
    paths = [p for p in glob.glob("lighthouse_experiments/*.py") if "__init__" not in p and "base.py" not in p]
    s = "python lighthouse_scripts/save_pairwise_imitation_data.py {} -m 1 --output_dir hp_runs/lighthouse --log_level error"
    cmd = " ; ".join([s.format(p) for p in paths])
    print(cmd)
    ```
    """

    # Get command line arguments
    args = get_args()

    # Define fixed parameters
    world_dim = 2
    world_radius = 15
    view_radii = list(range(1, 16, 2))
    use_experts = args.experiment.split(".")[-1] not in ["a2c", "ppo"]
    nrepeats = 25 if use_experts else 50  # Number of random seeds per experiment

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        max_processes_for_gpus = torch.cuda.device_count() * math.floor(
            gpu_memory / (2000 * (2 ** 20))
        )
    else:
        max_processes_for_gpus = 0

    nprocesses = (
        min(max_processes_for_gpus, math.floor(0.9 * mp.cpu_count()))
        if torch.cuda.is_available()
        else 1
    )

    gpu_ids = (
        [] if not torch.cuda.is_available() else list(range(torch.cuda.device_count()))
    )

    ptitle("Master (pairwise)")

    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Where to save data
    tsv_save_data_path = os.path.join(
        output_dir,
        "{}__{}_{}.tsv".format(
            args.experiment.replace(".py", "").replace(".", "_"),
            world_dim,
            world_radius,
        ),
    )

    # Get any experiment data already saved (e.g. from previous runs)
    if os.path.exists(tsv_save_data_path):
        df = pd.read_csv(tsv_save_data_path, sep="\t")
    else:
        df = pd.DataFrame(
            dict(
                view_radius=[],
                expert_view_radius=[],
                reached_near_optimal=[],
                avg_ep_length=[],
                train_steps=[],
                seed=[],
                start_time_str=[],
            )
        )

    # The experiments we've already run
    seen_tuples = set(
        zip(
            df["view_radius"],
            [None if np.isnan(x) else x for x in df["expert_view_radius"]],
            df["seed"],
        )
    )

    # Add experiments details into the `input_queue` but
    # don't include experiments we've already run.
    input_queue = mp.Queue()
    total_runs = 0
    for i, view_radius in enumerate(view_radii):
        for expert_view_radius in view_radii[i:] if use_experts else [None]:
            for seed in range(nrepeats):
                total_runs += 1
                t = (view_radius, expert_view_radius, seed)
                if t not in seen_tuples:
                    input_queue.put(t + (None,))

    output_queue = mp.Queue()

    # Create the subprocesses that run experiments.
    processes = []
    for i in range(min(nprocesses, total_runs - len(seen_tuples))):
        processes.append(
            mp.Process(
                target=iteratively_run_lighthouse_experiments,
                kwargs=dict(
                    process_id=i,
                    gpu_id=gpu_ids[i % len(gpu_ids)] if len(gpu_ids) != 0 else None,
                    args=args,
                    world_dim=world_dim,
                    world_radius=world_radius,
                    input_queue=input_queue,
                    output_queue=output_queue,
                    log_level=args.log_level,
                ),
            )
        )
        processes[-1].start()
        time.sleep(0.1)

    # Save experimental results from the subprocesses into a tsv file.
    os.makedirs(os.path.dirname(tsv_save_data_path), exist_ok=True)
    while len(seen_tuples) != total_runs:
        new_seen_tuple, run_data = output_queue.get()

        seen_tuples.add(new_seen_tuple[:-1])  # Don't include the learning rate

        df = df.append(run_data, ignore_index=True)

        df.to_csv(tsv_save_data_path, sep="\t", index=False)

    for p in processes:
        try:
            p.join(1)
        except Exception as _:
            pass

    print("Saving pairwise imitation data is done!")
