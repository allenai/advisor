import os
from pathlib import Path

import matplotlib.pyplot as plt

from allenact.utils.misc_utils import TABLEAU10_RGB

ADVISOR_TOP_LEVEL_DIR = os.path.abspath(os.path.dirname(Path(__file__)))

NICE_COLORS12_RGB = TABLEAU10_RGB + (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (128, 128, 128),
)

plt.rc("font", **{"family": "serif", "serif": ["CMU"]})
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

FIXED_ADVISOR_STR = r"ADV"

EXPERIMENT_STR_TO_LABEL_DICT = {
    "dagger_then_ppo": r"$\dagger \to$ PPO",
    "bc_then_ppo": r"BC$ \to$ PPO",
    "bc": r"BC",
    "dagger": r"DAgger $(\dagger)$",
    "ppo": r"PPO",
    "ppo_with_offpolicy": r"BC$^{\text{demo}} +$ PPO",
    "pure_offpolicy": r"BC$^{\text{demo}}$",
    "gail": r"GAIL",
    "bc_teacher_forcing": r"BC$^{\text{tf}=1}$",
    "bc_teacher_forcing_then_ppo": r"BC$^{\text{tf}=1} \to$ PPO",
    "bc_with_ppo": r"BC$+$PPO (static)",
    # ADVISOR
    "advisor": r"{}".format(FIXED_ADVISOR_STR),
    "dagger_then_advisor": r"$\dagger \to$ {}".format(FIXED_ADVISOR_STR),
    "ppo_with_offpolicy_advisor": r"ADV$^{\text{demo}} +$ PPO",
    "bc_teacher_forcing_then_advisor": r"BC$^{\text{tf}=1} \to$ ADV",
}

TYPE_TO_EXPERIMENT_STRS = {
    "rl": ["ppo"],
    "rl+il": [
        "bc_with_ppo",
        "bc_then_ppo",
        "dagger_then_ppo",
        "bc_teacher_forcing_then_ppo",
    ],
    "il": ["bc", "dagger", "bc_teacher_forcing"],
    "demos": ["pure_offpolicy", "ppo_with_offpolicy", "gail"],
    "advisor": [
        "advisor",
        "bc_teacher_forcing_then_advisor",
        "dagger_then_advisor",
        "ppo_with_offpolicy_advisor",
    ],
}

EXPERIMENT_STR_TO_TYPE = {
    v: k for k in TYPE_TO_EXPERIMENT_STRS for v in TYPE_TO_EXPERIMENT_STRS[k]
}

METHOD_ORDER = [
    v
    for k in ["rl", "il", "rl+il", "demos", "advisor"]
    for v in TYPE_TO_EXPERIMENT_STRS[k]
]

METHOD_TO_COLOR = {}
METHOD_TO_LINE_STYLE = {}
METHOD_TO_LINE_MARKER = {}
NICE_MARKERS = ("", "|", "x", "^")


def _init_method_to_dictionaries():
    for type_ind, type in enumerate(TYPE_TO_EXPERIMENT_STRS):

        # COLOR (based on type, "rl", "rl+il", etc)
        n = len(NICE_COLORS12_RGB)

        for method_ind, method in enumerate(TYPE_TO_EXPERIMENT_STRS[type]):

            METHOD_TO_COLOR[method] = NICE_COLORS12_RGB[
                (type_ind + (type_ind // n)) % n
            ]

            # STYLE
            METHOD_TO_LINE_STYLE[method] = ["solid", "dashed", "dashdot"][
                method_ind % 3
            ]

            # MARKER
            METHOD_TO_LINE_MARKER[method] = NICE_MARKERS[method_ind % len(NICE_MARKERS)]


_init_method_to_dictionaries()
