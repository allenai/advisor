import argparse
import glob
import multiprocessing as mp
import os

mp = mp.get_context("forkserver")

from projects.advisor.summarization_utils import create_comparison_hp_plots_from_tsv


def get_argument_parser():
    """Creates the argument parser."""

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="Random HP Search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dir",
        type=str,
        default="./hp_runs",
        help="directory in which the result tsv files were saved (default: './hp_runs')",
    )

    parser.add_argument(
        "--env_type",
        type=str,
        default="",
        help="which task to generate plots for, defaults to generating plots for all tasks.",
    )

    return parser


if __name__ == "__main__":
    """Summarize the information saved via `minigrid_random_hp_search` and save
    E[max()] plots. This script would summarize all tsv files saves in the
    `dir` directory, unless a `--env_name` flag directs it to a particular
    environment.

    Run this with the following commmand:
    `python projects/advisor/minigrid_and_pd_scripts/summarize_random_hp_search.py`
    """
    args = get_argument_parser().parse_args()

    dir = args.dir
    env_type = args.env_type

    # env_type = "AskForHelpSimpleCrossingOnce"
    # env_type = "PoisonedDoors"
    # env_type = "AskForHelpSimpleCrossingOnce"
    # env_type = "LavaCrossingCorruptExpertS15N7"

    highlight_best = True
    hide_labels = True

    overwrite = True

    paths = glob.glob(os.path.join(dir, "random_*.tsv"))
    paths = [p for p in paths if env_type.lower() in os.path.basename(p).lower()]

    processes = []
    for path in paths:
        print()
        print(os.path.basename(path))
        kwargs = dict(
            num_hp_evals_for_steps_plot=10,
            tsv_file_path=path,
            overwrite=overwrite,
            include_legend=False,
            highlight_best=highlight_best,
            hide_labels=hide_labels,
        )
        if len(paths) == 1:
            create_comparison_hp_plots_from_tsv(**kwargs)
        else:
            p = mp.Process(target=create_comparison_hp_plots_from_tsv, kwargs=kwargs)
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
