# Bridging the Imitation Gap by Adaptive Insubordination

This repository includes the code that can be used to reproduce the results of our
paper [Bridging the Imitation Gap by Adaptive Insubordination](https://arxiv.org/abs/2007.12173).

Our experiments were primarily run using the [AllenAct learning framework](https://github.com/allenai/allenact),
see [AllenAct.org](https://allenact.org/) for details and tutorials.


## 🔎 Table of contents

<div class="toc">
<ul>
<li><a href="#-installation">💻 Installation</a></li>
<li><a href="#-poisoneddoors-and-minigrid-experiments">🏁 PoisonedDoors and MiniGrid Experiments</a><ul>
<li><a href="#-generating-hyperparameter-plots-from-tsvs">📈 Generating Hyperparameter Plots from TSVs</a></li>
<li><a href="#-generating-tsvs">🧮 Generating TSVs</a></li>
</ul>
</li>
<li><a href="#-2d-lighthouse-experiments">💡🏠 2D-Lighthouse Experiments</a></li>
<li><a href="#-robothor-objectnav-experiments">🍅 RoboTHOR ObjectNav Experiments</a></li>
<li><a href="#-habitat-pointnav-and-mpe-coopnav-experiments">🚧 Habitat PointNav and MPE CoopNav Experiments</a></li>
<li><a href="#-citation">📜 Citation</a></li>
</ul>
</li>
</ul>
</div>

## 💻 Installation

Simply create a new python virtual environment and, within this virtual environment, run 
```
pip install -r requirements.txt
```

## 🏁 PoisonedDoors and MiniGrid Experiments
### 📈 Generating Hyperparameter Plots from TSVs
We have included 9 tsv files in the `hp_runs` directory corresponding to the 
PoisonedDoors task and the 8 MiniGrid (described in Appendix A.5 of the paper).
Each TSV file contains the results from training 50 hyperparameterized models for each of the 15 baselines included
in this work (see Table A.1 in our paper's appendix).

Using the following command the user can generate the plots for the task `WallCrossingCorruptExpertS25N10` i.e. `WC Corrupt (S25, N10)` of the submission (runtime: < 1min).

```bash
python minigrid_and_pd_scripts/summarize_random_hp_search.py --env_name AskForHelpLavaCrossingOnce
```

Running the above command will print something similar to:

```bash
random_hp_search_runs_AskForHelpLavaCrossingOnce.tsv
{'ppo': 50, 'bc': 50, 'dagger': 50, 'bc_teacher_forcing': 50, 'bc_with_ppo': 50, 'bc_then_ppo': 50, 'dagger_then_ppo': 50, 'bc_teacher_forcing_then_ppo': 50, 'pure_offpolicy': 50, 'ppo_with_offpolicy': 50, 'gail': 50, 'advisor': 50, 'bc_teacher_forcing_then_advisor': 50, 'dagger_then_advisor': 50, 'ppo_with_offpolicy_advisor': 50}
750
Starting expected max reward computations
Computing expected max reward for PPO
Computing expected max reward for BC
Computing expected max reward for DAgger $(\dagger)$
Computing expected max reward for BC$^{\text{tf}=1}$
Computing expected max reward for BC$+$PPO (static)
Computing expected max reward for BC$ \to$ PPO
Computing expected max reward for $\dagger \to$ PPO
Computing expected max reward for BC$^{\text{tf}=1} \to$ PPO
Computing expected max reward for BC$^{\text{demo}}$
Computing expected max reward for BC$^{\text{demo}} +$ PPO
Computing expected max reward for GAIL
Computing expected max reward for ADV
Computing expected max reward for BC$^{\text{tf}=1} \to$ ADV
Computing expected max reward for $\dagger \to$ ADV
Computing expected max reward for ADV$^{\text{demo}} +$ PPO
findfont: Font family ['serif'] not found. Falling back to DejaVu Sans.
Figure saved to hp_runs/neurips21_plots/AskForHelpLavaCrossingOnce/random_hp_search_runs_AskForHelpLavaCrossingOnce__train_steps_AskForHelpLavaCrossingOnce_reward.pdf
Figure saved to hp_runs/neurips21_plots/AskForHelpLavaCrossingOnce/random_hp_search_runs_AskForHelpLavaCrossingOnce__hpruns_AskForHelpLavaCrossingOnce_reward.pdf
```
indicating each of the 15 baselines and the number of hyperparameters searched per model. The last two line of this output
shows the location where the resulting plots were saved.

By simply omitting the `--env_name` flag i.e. using the following command, all the tasks would be processed (runtime: < 5min):

```bash
python minigrid_and_pd_scripts/summarize_random_hp_search.py
```

We have already included these pdfs, and the above commands would regenerate and overwrite them

### 🧮 Generating TSVs
We have included the scripts for generating the tsv files described above. These scripts take 18-24 hours on a 48 CPU,
4 NVIDIA T4 GPU machine (`g4dn.12xlarge` AWS instance).
The `minigrid_and_pd_scripts/random_hp_search.py` script achieves this. 
For `WallCrossingS25N10` task, this can be done with the following command (runtime: < 24 hours on `g4dn.12xlarge`):
```bash
python projects/advisor/minigrid_and_pd_scripts/random_hp_search.py \
    RUN \
    -m 1 \
    -b projects/advisor/minigrid_and_pd_experiments \
    --output_dir hp_runs \
    --log_level error \
    --env_name CrossingS25N10
```
As mentioned before, the above command runs 50 models for all baselines including those based solely on
expert demonstrations. For these to function, we need to save demonstrations for each of the tasks.
The `minigrid_and_pd_scripts/save_expert_demos.py` script handles this. For eg., to save demonstrations for
the `CrossingS25N10` task, use the following (runtime: < 15 mins on `g4dn.12xlarge`):
```bash
python minigrid_and_pd_scripts/save_expert_demos.py bc \
    -b minigrid_and_pd_experiments/ \
    -o minigrid_data/minigrid_demos \
    --env_name CrossingS25N10
```

## 💡🏠 2D-Lighthouse Experiments

For 2D-Lighthouse, we have included all the code under the `lighthouse_experiments` and `lighthouse_scripts`
directories. To reproduce Figures 6 and A.3 from our paper see the `lighthouse_scripts/save_pairwise_imitation_data.py` and
`lighthouse_scripts/summarize_pairwise_imitation_data.py` scripts. The TSV files produced by the `save_pairwise_imitation_data.py`
can be found in the `./hp_runs/lighthouse/pairwise_imitation_results` directory. The figures corresponding to these
TSV files are generated by the `summarize_pairwise_imitation_data.py`; we have generated these files for you, they
can be found in the `./hp_runs/lighthouse/metric_comparisons` directory.

## 🍅 RoboTHOR ObjectNav Experiments

Our RoboTHOR ObjectNav experiment configuration files can be found in the `objectnav_experiments/robothor` directory.
Note that these experiments only include the ADVISOR and BC+PPO baselines, our other two baselines (BC and PPO) 
are taken from the [ObjectNav baselines in AllenAct](https://github.com/allenai/allenact/tree/main/projects/objectnav_baselines).

## 🚧 Habitat PointNav and MPE CoopNav Experiments

Due to the nature of the Habitat and MPE experiments, we have conducted them in a separate codebase. If you are
interested in reproducing these results please submit an issue.

## 📜 Citation

If you use this work, please cite:

```text
@misc{advisor,
  author = {{Luca Weihs*} and {Unnat Jain*} and Jordi Salvador and Svetlana Lazebnik and Aniruddha Kembhavi and Alexander G Schwing},
  title = {Bridging the Imitation Gap by Adaptive Insubordination},
  year = {2020},
  journal = {arXiv preprint arXiv:2007.12173},
}

```