import ast
import hashlib
import json
import os
from collections import defaultdict
from typing import Tuple, Sequence, Dict, Optional, Union, Any, Set

import compress_pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
from filelock import FileLock

from allenact.utils.misc_utils import (
    bootstrap_max_of_subset_statistic,
    expected_max_of_subset_statistic,
    all_equal,
)
from minigrid_and_pd_scripts.compute_random_performance_for_task import (
    TASK_TO_RANDOM_PERFORMANCE,
)
from projects.advisor.advisor_constants import (
    METHOD_ORDER,
    METHOD_TO_COLOR,
    METHOD_TO_LINE_MARKER,
    EXPERIMENT_STR_TO_LABEL_DICT,
)
from projects.advisor.lighthouse_scripts.summarize_pairwise_imitation_data import (
    set_size,
)
from projects.advisor.minigrid_constants import ENV_NAMES_TO_TITLE

plt.rc("font", **{"family": "serif", "serif": ["CMU"], "size": 16})

plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

METRIC_TO_LABEL = {
    "reward": "Reward",
    "rewards": "Reward",
    "avg_ep_length": "Avg. Ep. Length",
    "success": "Success",
}


def unzip(xs):
    a = None
    n = None
    for x in xs:
        if n is None:
            n = len(x)
            a = [[] for _ in range(n)]
        for i, y in enumerate(x):
            a[i].append(y)
    return a


def add_columns_to_df(df):
    keys = ["alpha_start", "alpha_stop", "fixed_alpha", "lr", "tf_ratio"]
    for key in keys + ["pretty_label"]:
        df[key] = [None] * df.shape[0]

    def read_config_kwargs_str(config_kwargs_str):
        if config_kwargs_str == "" or config_kwargs_str is None:
            return {}
        elif isinstance(config_kwargs_str, Dict):
            return config_kwargs_str
        else:
            try:
                return json.loads(config_kwargs_str)
            except Exception:
                return ast.literal_eval(config_kwargs_str)

    df.loc[:, "config_kwargs"] = [
        read_config_kwargs_str(config_kwargs_str)
        for config_kwargs_str in df.loc[:, "config_kwargs_str"]
    ]

    for i in range(df.shape[0]):
        row = df.loc[i, :]
        config_kwargs: Dict[str, Any] = row["config_kwargs"]
        for key in keys:
            df.loc[i, key] = config_kwargs.get(key.upper(), None)

    for i in range(df.shape[0]):
        df.loc[i, "pretty_label"] = run_info_to_pretty_label(dict(df.loc[i, :]))

    return df


def plot_max_hp_curves(
    x_to_y_list: Sequence[Dict[Union[int, float], float]],
    x_to_bootstrap_ys_list: Sequence[Dict[Union[int, float], Sequence[float]]],
    method_labels: Sequence[str],
    colors: Sequence[Tuple[int, int, int]],
    line_styles: Optional[Sequence] = None,
    line_markers: Optional[Sequence] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    fig_size=(4, 4 * 3.0 / 5.0),
    save_path: Optional[str] = None,
    put_legend_outside: bool = True,
    include_legend: bool = False,
    performance_of_random_agent: Optional[float] = None,
    best_inds_to_highlight: Optional[Set] = None,
):
    """Plots E[max(metric | n hp runs)] curves.

    For more information on studying sensitivity of methods to
    hyperparameter tuning, refer to Dodge et al. EMNLP 2019
    https://arxiv.org/abs/1909.03004
    """
    line_styles = ["solid"] * len(colors) if line_styles is None else line_styles
    line_markers = [""] * len(colors) if line_markers is None else line_markers

    plt.grid(
        b=True,
        which="major",
        color=np.array([0.93, 0.93, 0.93]),
        linestyle="-",
        zorder=-2,
    )
    plt.minorticks_on()
    plt.grid(
        b=True,
        which="minor",
        color=np.array([0.97, 0.97, 0.97]),
        linestyle="-",
        zorder=-2,
    )
    ax = plt.gca()
    ax.set_axisbelow(True)
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if best_inds_to_highlight is None:
        best_inds_to_highlight = set(range(len(x_to_y_list)))

    xscaled = False
    for (
        index,
        (x_to_y, x_to_bootstrap_ys, method_label, color, line_style, line_marker,),
    ) in enumerate(
        zip(
            x_to_y_list,
            x_to_bootstrap_ys_list,
            method_labels,
            colors,
            line_styles,
            line_markers,
        )
    ):
        xvals = list(sorted(x_to_bootstrap_ys.keys()))
        points_list = [x_to_bootstrap_ys[x] for x in xvals]
        points = [x_to_y[x] for x in xvals]
        should_highlight = index in best_inds_to_highlight

        if max(xvals) > 1e3:
            xscaled = True
            xvals = [x / 1e6 for x in xvals]

        try:
            lower, _, upper = unzip(
                [np.percentile(points, [25, 50, 75]) for points in points_list]
            )

        except Exception as _:
            print(
                "Could not generate max_hp_curve for {}, too few points".format(
                    method_label
                )
            )
            continue

        if performance_of_random_agent is not None:
            xvals = [0] + xvals
            points = [performance_of_random_agent] + points
            lower = [performance_of_random_agent] + lower
            upper = [performance_of_random_agent] + upper

        plt.gca().fill_between(
            xvals,
            lower,
            upper,
            color=np.array(color + (25 if should_highlight else 0,)) / 255,
            zorder=1,
        )

        plot_kwargs = dict(
            lw=2.5,
            linestyle=line_style,
            marker=line_marker,
            markersize=8,
            markevery=4 if len(xvals) > 10 else 1,
            zorder=2,
        )
        label = (
            r"{}.{}".format(index + 1, "\ \ " if index + 1 < 10 else " ") + method_label
        )
        color = np.array(color + (255,)) / 255
        plt.plot([], [], label=label, color=color, **plot_kwargs)  # FOR LEGEND ONLY

        if not should_highlight:
            color = np.array(color)
            color[3] = 0.1
        plt.plot(xvals, points, color=color, **plot_kwargs)

    plt.title(title)
    plt.xlabel(xlabel + (r"(Millions)" if xscaled and len(xlabel) != 0 else r""))
    plt.ylabel(ylabel)
    plt.ticklabel_format(style="plain")
    plt.tight_layout()

    if include_legend:
        if put_legend_outside:
            ax = plt.gca()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        else:
            plt.legend()

    set_size(*fig_size)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(
            save_path, bbox_inches="tight",
        )
        plt.close()
        print(f"Figure saved to {save_path}")


def create_comparison_hp_plots_from_tsv(
    num_hp_evals_for_steps_plot: int,
    tsv_file_path: str,
    highlight_best: bool,
    overwrite=True,
    include_legend: bool = False,
    hide_labels: bool = False,
):
    assert os.path.exists(tsv_file_path)

    file_dir, file_name = os.path.split(tsv_file_path)

    with open(tsv_file_path, "r") as f:
        tsv_hash = str(hashlib.md5(f.read().encode()).hexdigest())

    df = pd.read_csv(tsv_file_path, sep="\t")

    df = add_columns_to_df(df)

    env_type_key = "env"
    assert (
        df[env_type_key] == df[env_type_key][0]
    ).all(), "env must be the same for all elements of df"

    task_name = df[env_type_key][0]

    del df[env_type_key]

    df = df.sort_values(by=["exp_type", "seed"])

    group_keys = ["exp_type"]

    df_grouped = df.groupby(by=group_keys)
    df_grouped_lists = df_grouped.agg(list)

    # One sort index, based on the first metric
    for metric_key in [
        "reward",
        # "success",
        # "avg_ep_length",
    ]:
        if not os.path.exists(file_dir):
            print("IN WRONG DIRECTORY.")
        else:
            plots_dir = os.path.join(file_dir, "neurips21_plots", task_name)
            os.makedirs(plots_dir, exist_ok=True)
            box_save_path = os.path.join(
                plots_dir,
                "{}__box_{}_{}.pdf".format(
                    file_name.replace(".tsv", ""), task_name, metric_key,
                ),
            )
            if (not overwrite) and os.path.exists(box_save_path):
                print(
                    "Plot {} exists and overwrite is `False`, skipping...".format(
                        box_save_path
                    )
                )
                continue

            tsv_summary_dir = os.path.join(file_dir, "neurips21_summaries")
            os.makedirs(tsv_summary_dir, exist_ok=True)
            tsv_summary_save_path = os.path.join(
                tsv_summary_dir, f"{metric_key}__all_results.tsv"
            )

            grouped_df_index = df_grouped_lists.index.to_frame(index=False)
            method_keys = list(grouped_df_index["exp_type"])
            sort_index = [
                ind
                for _, ind in sorted(
                    [
                        (METHOD_ORDER.index(method_key), sort_ind)
                        if method_key in METHOD_ORDER
                        else 1e6
                        for sort_ind, method_key in enumerate(method_keys)
                        if method_key in METHOD_ORDER
                    ]
                )
            ]
            colors = [
                METHOD_TO_COLOR.get(method_keys[ind], (0, 0, 0),) for ind in sort_index
            ]

            line_styles = None
            line_markers = [
                METHOD_TO_LINE_MARKER.get(method_keys[ind], "",) for ind in sort_index
            ]

            sorted_multi_index = [
                tuple(grouped_df_index.loc[ind, :]) for ind in sort_index
            ]

            sorted_multi_index = [
                x if len(x) != 1 else x[0] for x in sorted_multi_index
            ]

            result_lens = {
                multi_ind: len(df_grouped_lists.loc[multi_ind, metric_key])
                for multi_ind in sorted_multi_index
            }
            print(result_lens)
            print(sum(result_lens.values()))

            points_list = [
                list(
                    map(ast.literal_eval, df_grouped_lists.loc[multi_ind, metric_key],)
                )
                for multi_ind in sorted_multi_index
            ]

            exp_to_ckpt_training_steps_lists = [
                df_grouped_lists.loc[multi_ind, "train_steps"]
                for multi_ind in sorted_multi_index
            ]
            assert all(all_equal(l) for l in exp_to_ckpt_training_steps_lists)
            exp_ind_to_ckpt_training_steps = [
                ast.literal_eval(training_steps_list[0])
                for training_steps_list in exp_to_ckpt_training_steps_lists
            ]

            pretty_label_lists = [
                df_grouped_lists.loc[multi_ind, "pretty_label"]
                for multi_ind in sorted_multi_index
            ]
            assert all(all_equal(l) for l in pretty_label_lists)

            yticklabels = [l[0] for l in pretty_label_lists]

            subset_size_to_bootstrap_points_list = []
            subset_size_to_expected_mas_est_list = []
            ckpt_to_bootstrap_points_list = []
            ckpt_to_expected_mas_est_list = []
            print("Starting expected max reward computations")
            for i in range(len(points_list)):
                print(f"Computing expected max {metric_key} for {yticklabels[i]}")

                vals_per_ckpt_mat = np.array(
                    points_list[i]
                )  # each col corresponds to a checkpoint

                training_steps_inds_to_skip = []
                training_steps = exp_ind_to_ckpt_training_steps[i]

                cache_path = os.path.join(
                    plots_dir, "cache", f"{tsv_hash}_{i}_{metric_key}.pkl.gz"
                )
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                if os.path.exists(cache_path):
                    cache = compress_pickle.load(cache_path)

                    ckpt_to_expected_mas_est_list.append(
                        cache["ckpt_to_expected_mas_est"]
                    )
                    ckpt_to_bootstrap_points_list.append(
                        cache["ckpt_to_bootstrap_points"]
                    )
                    subset_size_to_expected_mas_est_list.append(
                        cache["subset_size_to_expected_mas_est"]
                    )
                    subset_size_to_bootstrap_points_list.append(
                        cache["subset_size_to_bootstrap_points"]
                    )
                else:
                    for j in range(len(training_steps) - 1):
                        # Skip some weird cases where checkpoints were saved too closely
                        if (training_steps[j + 1] - training_steps[j]) / training_steps[
                            -1
                        ] < 0.05:
                            training_steps_inds_to_skip.append(j)

                    ckpt_to_expected_mas_est_list.append(
                        {
                            training_steps: expected_max_of_subset_statistic(
                                vals_per_ckpt_mat[:, j], m=num_hp_evals_for_steps_plot
                            )
                            for j, training_steps in enumerate(training_steps)
                            if j not in training_steps_inds_to_skip
                        }
                    )
                    ckpt_to_bootstrap_points_list.append(
                        {
                            training_steps: bootstrap_max_of_subset_statistic(
                                vals_per_ckpt_mat[:, j],
                                m=num_hp_evals_for_steps_plot,
                                reps=500,
                                seed=j,
                            )
                            for j, training_steps in enumerate(training_steps)
                            if j not in training_steps_inds_to_skip
                        }
                    )

                    max_subset_size = len(points_list[i]) + 1 - 5
                    subset_size_to_expected_mas_est_list.append(
                        {
                            m: expected_max_of_subset_statistic(
                                vals_per_ckpt_mat[:, -1], m=m
                            )
                            for m in range(1, max_subset_size)
                        }
                    )
                    subset_size_to_bootstrap_points_list.append(
                        {
                            m: bootstrap_max_of_subset_statistic(
                                vals_per_ckpt_mat[:, -1], m=m, reps=500, seed=m
                            )
                            for m in range(1, max_subset_size)
                        }
                    )

                    cache = {}
                    cache["ckpt_to_expected_mas_est"] = ckpt_to_expected_mas_est_list[
                        -1
                    ]
                    cache["ckpt_to_bootstrap_points"] = ckpt_to_bootstrap_points_list[
                        -1
                    ]
                    cache[
                        "subset_size_to_expected_mas_est"
                    ] = subset_size_to_expected_mas_est_list[-1]
                    cache[
                        "subset_size_to_bootstrap_points"
                    ] = subset_size_to_bootstrap_points_list[-1]

                    compress_pickle.dump(cache, cache_path)

            color_to_best_val_and_index = defaultdict(lambda: (-float("inf"), -1))
            color_to_inds = defaultdict(lambda: [])
            for ind, c0 in enumerate(colors):
                color_to_inds[c0].append(ind)
                final_y = list(sorted(ckpt_to_expected_mas_est_list[ind].items()))[-1][
                    1
                ]
                if final_y > color_to_best_val_and_index[c0][0]:
                    color_to_best_val_and_index[c0] = (final_y, ind)
            best_inds_to_highlight = set(
                v[1] for v in color_to_best_val_and_index.values()
            )

            plot_max_hp_curves(
                x_to_y_list=ckpt_to_expected_mas_est_list,
                x_to_bootstrap_ys_list=ckpt_to_bootstrap_points_list,
                method_labels=yticklabels,
                xlabel=("Training Steps" if not hide_labels else ""),
                ylabel=(
                    f"Expected {METRIC_TO_LABEL[metric_key]}" if not hide_labels else ""
                ),
                colors=colors,
                line_styles=line_styles,
                line_markers=line_markers,
                fig_size=(3 * 1.05, 3 * 1.05),
                save_path=box_save_path.replace("_box_", "_train_steps_"),
                put_legend_outside=True,
                include_legend=include_legend,
                title=(ENV_NAMES_TO_TITLE[task_name] if not hide_labels else ""),
                performance_of_random_agent=TASK_TO_RANDOM_PERFORMANCE.get(
                    task_name, {}
                ).get(metric_key, None),
                best_inds_to_highlight=best_inds_to_highlight
                if highlight_best
                else None,
            )

            def save_expected_rewards_tsv(
                task_name: str,
                x_to_y_list: Sequence[Dict[Union[int, float], float]],
                method_labels: Sequence[str],
                save_path: str,
                grouped_inds_list: Sequence[Sequence[int]],
            ):
                def all_nearly_equal(seq):
                    s = seq[0]
                    return all(abs(s - ss) / min(s, ss) < 0.01 for ss in seq)

                with FileLock(save_path + ".lock"):
                    if os.path.exists(save_path):
                        df = pandas.read_csv(save_path, sep="\t")
                        assert list(df["method"]) == method_labels
                    else:
                        df = pandas.DataFrame(data={"method": method_labels})

                    assert all_nearly_equal(
                        [max(x_to_y.keys()) for x_to_y in x_to_y_list]
                    )

                    if task_name in df.columns:
                        del df[task_name]

                    values_at_end_of_training = [
                        x_to_y[max(x_to_y.keys())] for x_to_y in x_to_y_list
                    ]
                    df[task_name] = values_at_end_of_training

                    df = df.reindex(["method"] + list(sorted(df.columns[1:])), axis=1)
                    df.to_csv(save_path, sep="\t", index=False, float_format="%.2f")

                save_path = save_path.replace(".tsv", "_group.tsv")
                with FileLock(save_path + ".lock"):
                    grouped_method_labels = [
                        method_labels[inds[0]] for inds in grouped_inds_list
                    ]
                    if os.path.exists(save_path):
                        df = pandas.read_csv(save_path, sep="\t")
                        assert list(df["method"]) == grouped_method_labels
                    else:
                        df = pandas.DataFrame(data={"method": grouped_method_labels})
                    grouped_values_at_end_of_training = [
                        max(values_at_end_of_training[i] for i in inds)
                        for inds in grouped_inds_list
                    ]
                    df[task_name] = grouped_values_at_end_of_training
                    df = df.reindex(["method"] + list(sorted(df.columns[1:])), axis=1)
                    df.to_csv(save_path, sep="\t", index=False, float_format="%.2f")

            save_expected_rewards_tsv(
                task_name=ENV_NAMES_TO_TITLE[task_name],
                x_to_y_list=ckpt_to_expected_mas_est_list,
                method_labels=yticklabels,
                save_path=tsv_summary_save_path,
                grouped_inds_list=[
                    color_to_inds[k] for k in sorted(color_to_inds.keys())
                ],
            )

            plot_max_hp_curves(
                x_to_y_list=subset_size_to_expected_mas_est_list,
                x_to_bootstrap_ys_list=subset_size_to_bootstrap_points_list,
                method_labels=yticklabels,
                xlabel=("$N$" if not hide_labels else ""),
                ylabel=(
                    f"\emph{{Robust{METRIC_TO_LABEL[metric_key]}@$N$}}"
                    if not hide_labels
                    else ""
                ),
                colors=colors,
                line_styles=line_styles,
                line_markers=line_markers,
                fig_size=(3 * 1.05, 3 * 1.05),
                save_path=box_save_path.replace("_box_", "_hpruns_"),
                put_legend_outside=False,
                include_legend=False,
                title=(ENV_NAMES_TO_TITLE[task_name] if not hide_labels else ""),
                best_inds_to_highlight=best_inds_to_highlight
                if highlight_best
                else None,
            )


def run_info_to_pretty_label(run_info: Dict[str, Optional[Union[int, str, float]]]):
    exp_type = run_info["exp_type"]
    return EXPERIMENT_STR_TO_LABEL_DICT[exp_type]
