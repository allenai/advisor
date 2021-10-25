import copy
import glob
import os
import sys
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from advisor_constants import ADVISOR_TOP_LEVEL_DIR
from allenact.utils.misc_utils import TABLEAU10_RGB

plt.rc("font", **{"family": "serif", "serif": ["CMU"]})
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")


def boxplot(y, points, color, delta, lw, include_outliers=False):
    lq, median, uq = np.percentile(points, [25, 50, 75])
    iqr = uq - lq
    outliers = points[np.abs(points - median) > 1.5 * iqr]
    inliers = points[np.abs(points - median) <= 1.5 * iqr]

    min, max = np.percentile(inliers, [0, 100])

    delta = delta / 2
    plt.hlines(
        y, xmin=lq, xmax=uq, linewidth=lw, color=np.concatenate((color, [0.5]), axis=0)
    )
    plt.hlines(y, xmin=min, xmax=max, linewidth=lw / 4, color=color)
    plt.vlines(median, ymin=y - delta, ymax=y + delta, linewidth=lw / 2, color=color)

    if include_outliers and outliers.shape[0] != 0:
        plt.scatter(outliers, [y] * len(outliers), s=1, color=color)


def set_size(w, h, ax=None):
    """Set figure axis sizes.

    Taken from the answer in
    https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units

    w, h: width, height in inches
    """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def plot_boxplots(
    points_tensor: Sequence[Sequence[Sequence[float]]],
    optimal: Optional[np.ndarray] = None,
    title="",
    xlabel="",
    ylabel=r"View Radius ($i$)",
    yticklabels="default",
    save_path: Optional[str] = None,
    xlim: Optional[Union[Tuple[float, float], Tuple[str, float, float]]] = None,
    lw=2,
    colors: Optional[Sequence[Tuple[int, int, int]]] = None,
    fig_size=(4, 4 * 3.0 / 5.0),
    hline_after: Optional[int] = None,
    include_outliers: bool = False,
):
    nrows, ncols = len(points_tensor), len(points_tensor[0])
    assert all(len(row) == ncols for row in points_tensor)

    nboxes = np.logical_not(
        np.isnan([points_tensor[i][j][0] for i in range(nrows) for j in range(ncols)])
    ).sum()

    many_sub_boxes = nboxes > len(points_tensor)
    if many_sub_boxes:
        nboxes += len(points_tensor)
    yvalues = list(np.linspace(0, 1, num=nboxes + (hline_after is not None)))

    yticks = []
    yminorticks = []
    default_yticklabels = []

    plt.grid(
        b=True,
        axis="y",
        which="major",
        color=np.array([0.9, 0.9, 0.9]),
        linestyle="-",
        zorder=-2,
    )
    plt.minorticks_on()
    plt.grid(
        b=True,
        axis="y",
        which="minor",
        color=np.array([0.9, 0.9, 0.9]),
        linestyle="-",
        zorder=-2,
    )
    ax = plt.gca()
    ax.set_axisbelow(True)
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if hline_after is not None:
        plt.axhline(
            y=(yvalues[-(hline_after + 1)]), linewidth=lw / 8, color="black",
        )
        yvalues.pop(-(hline_after + 1))

    if ncols == 1:
        for y in yvalues:
            plt.axhline(y=y, linewidth=lw / 16, color="lightgrey", zorder=-1)

    for i in range(nrows):
        ys = []
        if many_sub_boxes and i != 0:
            yvalues.pop()
        for j in range(ncols):
            if not np.isnan(points_tensor[i][j][0]):
                ys.append(yvalues.pop())
                try:
                    boxplot(
                        # nrows - i + y_offsets.pop(),
                        ys[-1],
                        np.array(points_tensor[i][j]),
                        color=np.array(
                            TABLEAU10_RGB[i % 10] if colors is None else colors[i]
                        )
                        / 255,
                        delta=1 / (nboxes - 1),
                        lw=lw,
                        include_outliers=include_outliers,
                    )
                except Exception as _:
                    pass

        if len(ys) != 0:
            yticks.append(np.max(ys))
            yminorticks.extend(ys)
            default_yticklabels.append(i + 1)

        if optimal is not None and len(ys) != 0:
            plt.vlines(
                x=optimal[i],
                ymin=min(ys) - 1 / (2 * (nboxes - 1)),
                ymax=max(ys) + 1 / (2 * (nboxes - 1)),
                colors="grey",
                linewidth=0.5,
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel=ylabel)
    if yticklabels == "default":
        plt.yticks(yticks, labels=default_yticklabels)
        ax = plt.gca()
        ax.tick_params("y", which="major", direction="inout")
        ax.tick_params("y", which="minor", direction="in")
        ax.set_yticks(yminorticks, minor=True)
    else:
        plt.yticks(yticks, labels=list(yticklabels))
    plt.tight_layout()
    if xlim is not None:
        if xlim[0] != "proportional":
            plt.xlim(xmin=xlim[0], xmax=xlim[1])
        else:
            plt.tight_layout()
            _, right = plt.xlim()
            _, xmin, xmax = xlim
            right = max(xmin, min(xmax, right))
            plt.xlim(xmin=0, xmax=right)

            fig_size = (fig_size[0] * (1 - (xmax - right) / xmax), fig_size[1])

    set_size(*fig_size)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(
            save_path, bbox_inches="tight",
        )
        plt.close()
        print(f"Figure saved to {save_path}")


def plot_means_and_CIs(
    means_mat: np.ndarray,
    ci_lows_mat: np.ndarray,
    ci_highs_mat: np.ndarray,
    optimal: np.ndarray,
    title="",
    xlabel="",
    ylabel=r"View Radius ($i$)",
    yticklabels="default",
    save_path: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
):
    # fig = plt.figure(figsize=tuple(2 * np.array((5, 3))))

    results_per_row = []
    nrows, ncols = means_mat.shape[0], means_mat.shape[1]
    for i in range(means_mat.shape[0]):
        results_for_row = []
        for j in range(means_mat.shape[1]):
            if not np.isnan(means_mat[i, j]):
                results_for_row.append(
                    (means_mat[i, j], ci_lows_mat[i, j], ci_highs_mat[i, j])
                )
        results_per_row.append(results_for_row)

    for i in range(len(results_per_row)):
        if optimal is not None:
            plt.vlines(
                x=optimal[i],
                ymin=nrows - i - 0.5,
                ymax=nrows - i + 0.5,
                linestyles="dashed",
                colors="grey",
            )

        means = [t[0] for t in results_per_row[i]]
        ci_lows = [t[1] for t in results_per_row[i]]
        ci_highs = [t[2] for t in results_per_row[i]]

        nmeans = len(means)
        y_offsets = -(
            np.linspace(0.0 + 1 / (nmeans + 1), 1.0, num=nmeans, endpoint=False) - 0.5
        )
        plt.scatter(means, nrows - i + y_offsets)

        plt.hlines(nrows - i + y_offsets, xmin=ci_lows, xmax=ci_highs)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel=ylabel)
    if yticklabels == "default":
        plt.yticks(range(1, nrows + 1), labels=nrows - np.array(range(nrows)))
    else:
        plt.yticks(range(1, nrows + 1), labels=list(reversed(yticklabels)))
    plt.tight_layout()
    if xlim is not None:
        plt.xlim(xmin=xlim[0], xmax=xlim[1])

    if save_path is None:
        plt.show()
    else:
        plt.savefig(
            save_path, bbox_inches="tight",
        )
        plt.close()
        print(f"Figure saved to {save_path}")


if __name__ == "__main__":
    lighthouse_save_dir = os.path.join(ADVISOR_TOP_LEVEL_DIR, "hp_runs", "lighthouse")
    tsv_dir = os.path.join(lighthouse_save_dir, "pairwise_imitation_results")

    if len(sys.argv) > 1:
        file_paths = [os.path.join(tsv_dir, "{}__2_15.tsv".format(sys.argv[1]))]
    else:
        file_paths = glob.glob(os.path.join(tsv_dir, "*.tsv"))

    for file_path in file_paths:

        df = pd.read_csv(file_path, sep="\t")
        df["early_exit"] = 1.0 * (df["train_steps"] < 300000)

        bernoulli_keys = [
            key
            for key in df.keys()
            if np.all(np.logical_or(df[key] == 1.0, df[key] == 0.0))
        ]

        nan_expert_view_radii = np.isnan(df["expert_view_radius"])
        assert (not nan_expert_view_radii.any()) or nan_expert_view_radii.all()

        if nan_expert_view_radii.any():
            df["expert_view_radius"] = 1

        df_grouped_mean = df.groupby(by=["view_radius", "expert_view_radius"]).mean()
        df_grouped_count = df.groupby(by=["view_radius", "expert_view_radius"]).count()
        df_grouped_std = df.groupby(
            by=["view_radius", "expert_view_radius"]
        ).std() / np.sqrt(df_grouped_count)

        results_shape = (
            int(np.max(df["view_radius"])),
            int(np.max(df["expert_view_radius"])),
        )
        key_to_means: Dict[str, np.ndarray] = defaultdict(
            lambda: np.full(results_shape, fill_value=float("nan"),)
        )
        key_to_stds = copy.deepcopy(key_to_means)
        key_to_counts = copy.deepcopy(key_to_means)

        key_to_point_tensors: Dict[str, np.ndarray] = defaultdict(
            lambda: np.full(
                (*results_shape, df_grouped_count.max()["avg_ep_length"],),
                fill_value=float("nan"),
            )
        )

        for view_radius, expert_view_radius in list(df_grouped_mean.index):

            means = df_grouped_mean.loc[(view_radius, expert_view_radius)]
            stds = df_grouped_std.loc[(view_radius, expert_view_radius)]
            counts = df_grouped_count.loc[(view_radius, expert_view_radius)]

            for key in means.keys():
                if key == "seed":
                    continue

                key_to_means[key][
                    int(view_radius) - 1, int(expert_view_radius) - 1
                ] = means[key]
                key_to_stds[key][
                    int(view_radius) - 1, int(expert_view_radius) - 1
                ] = stds[key]
                key_to_counts[key][
                    int(view_radius) - 1, int(expert_view_radius) - 1
                ] = counts[key]

                points = np.array(
                    df.query(
                        "view_radius=={} and expert_view_radius=={}".format(
                            view_radius, expert_view_radius
                        )
                    )[key]
                )
                key_to_point_tensors[key][
                    int(view_radius) - 1, int(expert_view_radius) - 1, : points.shape[0]
                ] = points

        key_to_ci_low = {}
        key_to_ci_high = {}
        for key in key_to_means:
            if key == "seed":
                continue
            means = key_to_means[key]
            stds = key_to_stds[key]
            counts = key_to_counts[key]

            key_to_ci_low[key] = np.zeros(means.shape)
            key_to_ci_high[key] = np.zeros(means.shape)
            low = key_to_ci_low[key]
            high = key_to_ci_high[key]

            for i in range(means.shape[0]):
                for j in range(means.shape[1]):
                    mean = means[i, j]
                    count = counts[i, j]
                    std = stds[i, j]

                    if not np.isnan(mean):
                        if key in bernoulli_keys:
                            low[i, j], high[i, j] = proportion_confint(
                                count=mean * count,
                                nobs=count,
                                alpha=0.05,
                                method="jeffreys",
                            )
                        else:
                            low[i, j], high[i, j] = mean + 1.96 * std * np.array(
                                [-1, 1]
                            )
                    else:
                        low[i, j] = np.nan
                        high[i, j] = np.nan

        save_plot_dir = "pairwise_plots_" + "_".join(
            file_path.replace(".tsv", "").split("_")[-2:]
        )
        for key in [
            "avg_ep_length",
        ]:
            save_dir = os.path.join(lighthouse_save_dir, "metric_comparisons")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                "{}__{}.pdf".format(
                    key, os.path.split(file_path)[-1].replace(".tsv", "")
                ),
            )
            optimal = (
                np.nanmin(key_to_means["optimal"], axis=1)
                if key == "avg_ep_length"
                else None
            )
            if key == "train_steps":
                xlim = (0, None)
            elif key == "avg_ep_length":
                xlim = ("proportional", 110, 850)
            else:
                xlim = None

            if key in bernoulli_keys:
                plot_means_and_CIs(
                    means_mat=key_to_means[key],
                    ci_lows_mat=key_to_ci_low[key],
                    ci_highs_mat=key_to_ci_high[key],
                    optimal=optimal,
                    xlabel=key.replace("_", " ").title(),
                    save_path=save_path,
                    xlim=xlim,
                )
            else:
                fig_size = (3, 3 * 3.0 / 5.0)

                # THINNING
                if key_to_point_tensors[key].shape[0] == 15:
                    for i in [13, 15]:  # [3,7,11,15]:
                        key_to_point_tensors[key][i - 1, :, :] = np.nan

                plot_boxplots(
                    points_tensor=key_to_point_tensors[key],
                    optimal=optimal,
                    xlabel=key.replace("_", " ").title(),
                    save_path=save_path,
                    xlim=xlim,
                    fig_size=fig_size,
                )
