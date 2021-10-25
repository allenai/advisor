from allenact_plugins.babyai_plugin.scripts.truncate_expert_demos import (
    make_small_demos,
)
from projects.advisor.minigrid_constants import MINIGRID_EXPERT_TRAJECTORIES_DIR

if __name__ == "__main__":
    make_small_demos(MINIGRID_EXPERT_TRAJECTORIES_DIR)
