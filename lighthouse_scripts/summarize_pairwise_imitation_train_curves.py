import glob
import json
import os
import traceback
import warnings
from collections import defaultdict
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.framework.errors_impl import DataLossError
from tensorflow.python.summary.summary_iterator import summary_iterator

from allenact.utils.misc_utils import TABLEAU10_RGB

_color_iter = iter(TABLEAU10_RGB)
HEAD_NAME_TO_COLOR: Dict[str, Tuple[int, int, int]] = defaultdict(
    lambda: next(_color_iter)
)

VISUALIZATION = True
GENERATE_JSONS = True
OVERWRITE = True

if __name__ == "__main__":
    # TODO: Allow for changing `experiment_tag` via command line arguments.

    # experiment_tag = "advisor_ppo"
    experiment_tag = "dagger_then_ppo"

    cur_wd = os.getcwd()
    assert os.path.basename(cur_wd) == "advisor"
    saved_data_base_dir = "lighthouse_pairwise_training"
    log_dir = os.path.join(cur_wd, saved_data_base_dir, "tb")

    saved_processed_output_dir = os.path.join(
        cur_wd, saved_data_base_dir, "logs_processed_to_jsons"
    )
    os.makedirs(saved_processed_output_dir, exist_ok=True)

    if GENERATE_JSONS:
        tag_to_train_or_valid_and_nice_key = {"train/ep_length": ("train", "ep_length")}

        tsv_file_path = os.path.join(
            saved_data_base_dir, "{}__2_10.tsv".format(experiment_tag)
        )
        df = pd.read_csv(tsv_file_path, sep="\t")

        view_radii = set(int(x) if x is not None else x for x in df["view_radius"])
        expert_view_radii = set(
            int(x) if x is not None else x for x in df["expert_view_radius"]
        )

        for view_radius in view_radii:
            for expert_view_radius in expert_view_radii:
                log_save_path = os.path.join(
                    saved_processed_output_dir,
                    "{}__{}_{}__info.json".format(
                        experiment_tag, view_radius, expert_view_radius
                    ),
                )

                if not OVERWRITE and os.path.exists(log_save_path):
                    print(
                        "{} already exists and we're not overwritting, skipping...".format(
                            os.path.basename(log_save_path)
                        )
                    )
                    continue

                subdf = df.query(
                    "view_radius == {} & expert_view_radius == {}".format(
                        view_radius, expert_view_radius
                    )
                )

                ids = list(subdf["start_time_str"])

                event_paths = [
                    p
                    for id in ids
                    for p in glob.glob(os.path.join(log_dir, "*", id, "events.out*"))
                ]

                if len(event_paths) == 0:
                    continue

                save_data_per_event_train: Dict[
                    str, List[List[Tuple[int, float]]]
                ] = defaultdict(lambda: [])

                for event_path in event_paths:
                    save_data_per_metric_train: Dict[
                        str, List[Tuple[int, float]]
                    ] = defaultdict(lambda: [])
                    try:
                        for summary in summary_iterator(event_path):
                            try:
                                step = summary.step
                                tag_and_value = summary.summary.value[0]
                                metric_id = tag_and_value.tag

                                if metric_id not in tag_to_train_or_valid_and_nice_key:
                                    continue

                                value = tag_and_value.simple_value

                                (
                                    train_or_val,
                                    nice_tag,
                                ) = tag_to_train_or_valid_and_nice_key[metric_id]
                                assert train_or_val == "train"

                                save_data_per_metric_train[nice_tag].append(
                                    (step, value)
                                )

                            except Exception as _:
                                pass
                    except DataLossError as _:
                        warnings.warn("Data loss error in {}".format(event_path))

                    for k, v in save_data_per_metric_train.items():
                        save_data_per_event_train[k].append(v)

                with open(log_save_path, "w",) as f:
                    json.dump(save_data_per_event_train, f)

    if VISUALIZATION:
        for file_path in glob.glob(os.path.join(saved_processed_output_dir, "*.json")):
            plot_save_dir = os.path.join(cur_wd, "pairwise_plots", "train_curves")
            os.makedirs(plot_save_dir, exist_ok=True)
            save_plot_path = os.path.join(
                plot_save_dir,
                "{}.pdf".format(
                    "__".join(os.path.basename(file_path).split("__")[:-1])
                ),
            )

            if not OVERWRITE and os.path.exists(save_plot_path):
                print(
                    "{} already exists and we're not overwritting, skipping...".format(
                        os.path.basename(save_plot_path)
                    )
                )
                continue

            figsize = (4, 3)
            overwrite = False

            with open(file_path, "r") as f:
                metric_values = json.load(f)

            plt.figure(figsize=figsize)
            try:
                for step_and_ep_length_list in metric_values["ep_length"]:

                    x = [step for step, _ in step_and_ep_length_list]
                    y = [ep_length for _, ep_length in step_and_ep_length_list]

                    plt.plot(
                        x, y, color=(0.0, 0.0, 1.0, 0.2),
                    )

                plt.xlim(9e3, 4e5)

                plt.xlabel("Train steps")
                plt.ylabel("Episode Length")
                plt.xscale("log")
                plt.savefig(save_plot_path, bbox_inches="tight")
            except Exception as e:
                traceback.print_tb(e.__traceback__)
                print("Continuing")
            finally:
                plt.close()
