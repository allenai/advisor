import os
from pathlib import Path

MINIGRID_EXPERT_TRAJECTORIES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(Path(__file__)), "minigrid_data", "minigrid_demos")
)
MINIGRID_ENV_NAMES_SUPPORTED = (
    "CrossingS25N10",  # LavaCrossing (S25, N10)
    "WallCrossingS25N10",  # WallCrossing (S25, N10)
    "AskForHelpSimpleCrossing",  # WC Faulty Switch (S15, N7)
    "WallCrossingCorruptExpertS25N10",  # WC Corrupt (S25, N10)
    "AskForHelpLavaCrossingSmall",  # LC Faulty Switch (S9, N4)
    "AskForHelpSimpleCrossingOnce",  # WC Once Switch (S15, N7)
    "AskForHelpLavaCrossingOnce",  # LC Once Switch (S15, N7)
    "LavaCrossingCorruptExpertS15N7",  # LC Corrupt (S15, N7)
)

ENV_NAMES_TO_TITLE = {
    "CrossingS25N10": r"$\textsc{LavaCrossing (LC)}$",
    "WallCrossingS25N10": r"$\textsc{WallCrossing (WC)}$",
    "AskForHelpSimpleCrossing": r"$\textsc{WC Faulty Switch}$",
    "WallCrossingCorruptExpertS25N10": r"$\textsc{WC Corrupt}$",
    "AskForHelpLavaCrossingSmall": r"$\textsc{LC Faulty Switch}$",
    "AskForHelpSimpleCrossingOnce": r"$\textsc{WC Once Switch}$",
    "AskForHelpLavaCrossingOnce": r"$\textsc{LC Once Switch}$",
    "LavaCrossingCorruptExpertS15N7": r"$\textsc{LC Corrupt}$",
    "PoisonedDoors": r"$\textsc{PoisonedDoors}$",
}


def demos_exist_for_env(env_name: str):
    if env_name.lower().strip() == "poisoneddoors":
        return True
    return os.path.exists(
        os.path.join(MINIGRID_EXPERT_TRAJECTORIES_DIR, f"MiniGrid-{env_name}-v0.pkl")
    )
