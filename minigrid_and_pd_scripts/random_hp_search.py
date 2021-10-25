import itertools
import json
import math
import os
import queue
import time
import warnings
from typing import Dict, List, Optional, Any

import canonicaljson
import torch
import torch.multiprocessing as mp

from allenact.algorithms.onpolicy_sync.runner import OnPolicyRunner
from allenact.main import init_logging, load_config, get_argument_parser
from allenact.utils.misc_utils import rand_float
from allenact.utils.system import get_logger, update_log_level
from projects.advisor.minigrid_and_pd_experiments.base import BaseExperimentConfig
from projects.advisor.minigrid_constants import (
    MINIGRID_ENV_NAMES_SUPPORTED,
    demos_exist_for_env,
)

mp = mp.get_context("forkserver")
from setproctitle import setproctitle as ptitle
import pandas as pd
import numpy as np


def generate_random_lrs_tf_ratio_and_alphas(nsamples: int):
    np.random.seed(1)
    lr_samples = np.exp(rand_float(math.log(1e-4), math.log(0.5), nsamples))

    np.random.seed(2)
    tf_ratios = rand_float(0.1, 0.9, nsamples)

    np.random.seed(3)
    fixed_alphas = [np.random.choice([4.0, 8.0, 16.0, 32.0]) for _ in range(nsamples)]

    return lr_samples, tf_ratios, fixed_alphas


def iteratively_run_experiments(
    process_id: int,
    gpu_id: Optional[int],
    args,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    log_level: str,
    test_seed_offset: int = 0,
):
    """Iteratively train and test explore/combination models under different
    training regimes.

    This function is very similar to the `iteratively_run_lighthouse_experiments` function except
    that rather than training with different levels of supervision, here we only have one
    level of supervision and instead it's the training regime (e.g. ppo v.s. dagger) that is
    allowed to change based on the values in the `input_queue`.

    See `iteratively_run_lighthouse_experiments` for detailed documentation.
    """
    ptitle("({}) Create Iterative Experiment Runner".format(process_id))

    init_logging(log_level)

    def log_info(msg):
        update_log_level(logger=get_logger(), human_log_level="info")
        get_logger().info(msg)
        update_log_level(logger=get_logger(), human_log_level=log_level)

    try:
        while True:
            task_name, experiment_str, config_kwargs, seed = input_queue.get(timeout=1)

            args.experiment = experiment_str
            args.config_kwargs = json.dumps(
                {
                    "task_name": task_name,
                    "GPU_ID": gpu_id,
                    "TEST_SEED_OFFSET": test_seed_offset,
                    **config_kwargs,
                }
            )
            args.disable_tensorboard = True
            args.disable_config_saving = True

            cfg: BaseExperimentConfig = load_config(args)[0]  # type: ignore

            # assert agent_view_size % 2 == 1
            optimal_ave_ep_length = cfg.task_info().get("optimal_ave_ep_length")

            log_info(
                f"Running training with (env, exp, config_kwargs, seed) ="
                f" ({task_name}, {experiment_str}, {config_kwargs}, {seed})"
            )

            runner = OnPolicyRunner(
                config=cfg,
                output_dir=args.output_dir,
                loaded_config_src_files=None,
                seed=args.seed,
                mode="train",
                mp_ctx=mp,
                disable_tensorboard=args.disable_tensorboard,
                disable_config_saving=args.disable_config_saving,
            )

            train_start_time_str = runner.start_train(
                max_sampler_processes_per_worker=1,
                save_ckpt_after_every_pipeline_stage=False,
            )

            ckpt_dir = runner.checkpoint_dir(
                start_time_str=train_start_time_str, create_if_none=False
            )

            log_info(
                f"Running testing with (env, exp, config_kwargs, seed) ="
                f" ({task_name}, {experiment_str}, {config_kwargs}, {seed})"
            )

            runner.mode = "test"
            test_results = runner.start_test(
                checkpoint_path_dir_or_pattern=ckpt_dir,
                max_sampler_processes_per_worker=1,
            )
            runner.close()

            log_info(
                f"Testing complete for (minigrid, exp, config_kwargs, seed) ="
                f" ({task_name}, {experiment_str}, {config_kwargs}, {seed})"
            )

            output_data = {
                "exp_type": experiment_str,
                "env": task_name,
                "config_kwargs_str": canonicaljson.encode_canonical_json(
                    config_kwargs
                ).decode("utf-8"),
                "reward": [float(tr["reward"]) for tr in test_results],
                "avg_ep_length": [float(tr["ep_length"]) for tr in test_results],
                "train_steps": [float(tr["training_steps"]) for tr in test_results],
                "seed": seed,
                "lr": cfg.exp_params.LR,
                "extra_tag": cfg.extra_tag(),
            }
            if optimal_ave_ep_length is not None:
                output_data.update(
                    {
                        "reached_near_optimal": [
                            1 * (tr["ep_length"] < optimal_ave_ep_length * 1.1)
                            for tr in test_results
                        ],
                        "optimal_avg_ep_length": optimal_ave_ep_length,
                    }
                )

            for k in test_results[0]:
                if any(
                    metric_str in k
                    for metric_str in [
                        "success",
                        "found_goal",
                        "max_comb_correct",
                        "chose_",
                        "opened_",
                    ]
                ):
                    output_data[k] = [float(tr.get(k, np.nan)) for tr in test_results]

            output_queue.put((seed, output_data,))
    except queue.Empty:
        log_info("Queue empty for worker {}, exiting.".format(process_id))


if __name__ == "__main__":
    """Sample (equally) over hyperparams for each baseline. Aggregate
    information in a tsv. This leads to a TSV with `nsamples` times
    `number_of_baselines` If offpolicy baselines are to be run,
    `demos_exist_for_env` should be able to find demos for the environment
    being hp searched for.

    Run this with the following command.

    Command:
    ```
    python projects/advisor/minigrid_and_pd_scripts/random_hp_search.py \
        RUN \
        -m 1 \
        -b projects/advisor/minigrid_and_pd_experiments \
        --output_dir hp_runs \
        --log_level error \
        --env_name CrossingS25N10
    ```
    """

    parser = get_argument_parser()
    parser.add_argument(
        "--env_name", type=str, required=True,
    )
    args = parser.parse_args()

    nsamples = 50

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        max_processes_for_gpus = torch.cuda.device_count() * math.floor(
            gpu_memory / (1300 * (2 ** 20))
        )
    else:
        max_processes_for_gpus = 0

    nprocesses = (
        min(max_processes_for_gpus, math.floor(0.9 * mp.cpu_count()))
        if torch.cuda.is_available()
        else 4
    )
    gpu_ids = (
        [] if not torch.cuda.is_available() else list(range(torch.cuda.device_count()))
    )

    lr_samples, tf_ratios, fixed_alphas = generate_random_lrs_tf_ratio_and_alphas(
        nsamples=nsamples
    )

    lr_cfg_kwargs = [{"LR": lr} for lr in lr_samples]
    tf_ratio_cfg_kwargs = [{"TF_RATIO": ratio} for ratio in tf_ratios]
    fixed_alpha_cfg_kwargs = [
        {"FIXED_ALPHA": fixed_alpha} for fixed_alpha in fixed_alphas
    ]

    lr_tf_ratio_cfg_kwargs = [
        {**a, **b} for a, b in zip(lr_cfg_kwargs, tf_ratio_cfg_kwargs)
    ]

    fixed_advisor_cfg_kwargs = [
        {**a, **b} for a, b in zip(lr_cfg_kwargs, fixed_alpha_cfg_kwargs)
    ]

    dagger_fixed_advisor_cfg_kwargs = [
        {**a, **b, **c}
        for a, b, c in zip(lr_cfg_kwargs, tf_ratio_cfg_kwargs, fixed_alpha_cfg_kwargs)
    ]

    experiment_types_and_cfg_kwargs: Dict[str, List[Dict[str, Any]]] = {
        "bc_teacher_forcing_then_ppo": lr_tf_ratio_cfg_kwargs,
        "bc_teacher_forcing_then_advisor_fixed_alpha_different_head_weights": dagger_fixed_advisor_cfg_kwargs,
        "bc_teacher_forcing_then_advisor": dagger_fixed_advisor_cfg_kwargs,
        "bc": lr_cfg_kwargs,
        "dagger": lr_tf_ratio_cfg_kwargs,
        "ppo": lr_cfg_kwargs,
        "advisor_fixed_alpha_different_heads": fixed_advisor_cfg_kwargs,
        "advisor": fixed_advisor_cfg_kwargs,
        "bc_teacher_forcing": lr_cfg_kwargs,
        "dagger_then_advisor_fixed_alpha_different_head_weights": dagger_fixed_advisor_cfg_kwargs,
        "dagger_then_advisor": dagger_fixed_advisor_cfg_kwargs,
        "dagger_then_ppo": lr_tf_ratio_cfg_kwargs,
        "bc_then_ppo": lr_tf_ratio_cfg_kwargs,
        "bc_with_ppo": lr_cfg_kwargs,
        "gail": lr_cfg_kwargs,
    }

    if demos_exist_for_env(args.env_name):
        experiment_types_and_cfg_kwargs.update(
            {
                "ppo_with_offpolicy_advisor_fixed_alpha_different_heads": fixed_advisor_cfg_kwargs,
                "ppo_with_offpolicy_advisor": fixed_advisor_cfg_kwargs,
                "ppo_with_offpolicy": lr_cfg_kwargs,
                "pure_offpolicy": lr_cfg_kwargs,
            }
        )
    else:
        warnings.warn(
            "No demos found for {}, will not run off policy methods.".format(
                args.env_name
            )
        )

    assert args.env_name in ("PoisonedDoors",) + MINIGRID_ENV_NAMES_SUPPORTED

    # Currently, saving data for one task at a time
    task_names = [args.env_name]
    exp_type_and_cfg_kwargs_list = []

    for exp_type, cfg_kwargs_variants in experiment_types_and_cfg_kwargs.items():
        if len(cfg_kwargs_variants) == 0:
            cfg_kwargs_variants = [None]
        for seed, cfg_kwargs in enumerate(cfg_kwargs_variants):
            exp_type_and_cfg_kwargs_list.append((exp_type, cfg_kwargs, seed))

    ptitle("Master ({})".format(" and ".join(task_names)))

    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    assert len(task_names) == 1
    matrix_save_data_path = os.path.join(
        output_dir, "random_hp_search_runs_{}.tsv".format(task_names[0]),
    )

    if os.path.exists(matrix_save_data_path):
        df = pd.read_csv(matrix_save_data_path, sep="\t")
        df = df.where(pd.notnull(df), None)
        df["config_kwargs_str"] = df["config_kwargs_str"].astype(str)
    else:
        df = pd.DataFrame(
            dict(
                env=[],
                exp_type=[],
                config_kwargs_str=[],
                success=[],
                reached_near_optimal=[],
                avg_ep_length=[],
                train_steps=[],
                found_goal=[],
                max_comb_correct=[],
                seed=[],
                extra_tag=[],
                lr=[],
            )
        )

    seen_tuples = set(
        zip(df["env"], df["exp_type"], df["config_kwargs_str"], df["seed"])
    )
    all_tuples_to_train_set = set()

    input_queue = mp.Queue()
    input_queue_names = []
    total_runs = 0

    for env, (exp_type, cfg_kwargs, seed) in list(
        itertools.product(task_names, exp_type_and_cfg_kwargs_list)
    ):
        total_runs += 1
        t = (env, exp_type, cfg_kwargs, seed)

        # df loads cfg_kwargs as a string
        t_for_matching = (
            env,
            exp_type,
            canonicaljson.encode_canonical_json(cfg_kwargs).decode("utf-8"),
            seed,
        )
        all_tuples_to_train_set.add(t_for_matching)
        if t_for_matching not in seen_tuples:
            input_queue.put(t)
            input_queue_names.append(str((t[1], t[3])))

    seen_tuples = seen_tuples & all_tuples_to_train_set
    print("Queue:" + "\n".join(input_queue_names))
    output_queue = mp.Queue()

    print(
        "{} (of {}) experiments already completed! Running the rest.".format(
            len(seen_tuples), total_runs
        )
    )

    processes = []
    nprocesses = min(nprocesses, total_runs - len(seen_tuples))
    print(f"Starting {args.env_name} HP Search with {nprocesses} processes.")
    for i in range(nprocesses):
        processes.append(
            mp.Process(
                target=iteratively_run_experiments,
                kwargs=dict(
                    process_id=i,
                    gpu_id=gpu_ids[i % len(gpu_ids)] if len(gpu_ids) != 0 else None,
                    args=args,
                    input_queue=input_queue,
                    output_queue=output_queue,
                    log_level=args.log_level,
                ),
            )
        )
        processes[-1].start()
        time.sleep(0.1)

    while len(seen_tuples) != total_runs:
        try:
            output_seed, run_data = output_queue.get(timeout=120)
        except queue.Empty as _:
            print(
                f"{120} seconds passed without any experiment completing, continuing wait..."
            )
            continue

        seen_tuple = (
            run_data["env"],
            run_data["exp_type"],
            run_data["config_kwargs_str"],
            output_seed,
        )
        seen_tuples.add(seen_tuple)

        df = df.append(run_data, ignore_index=True)

        df.to_csv(matrix_save_data_path, sep="\t", index=False)

        print(f"Run {seen_tuple} saved, {len(seen_tuples)}/{total_runs} complete.")

    for p in processes:
        try:
            p.join(1)
        except Exception as _:
            pass

    print("Saving HP data is done!")
