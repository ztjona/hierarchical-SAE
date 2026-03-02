# -*- coding: utf-8 -*-

#

"""
Python 3
21 / 10 / 2025
@author: z_tjona


"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

"Either mathematics is too big for the human mind or the human mind is more than a machine."
-Kurt Godël
"""

from quartopy import play_games
from bot.CNN_bot import Quarto_bot
from bot.CNN_F_bot import Quarto_bot as F_bot
from QuartoRL import run_contest
import os
import glob
from tqdm.auto import tqdm
from pathlib import Path
from pprint import pprint


NUM_MATCHES = 30
# DIR_CKPTs = "CHECKPOINTS//QuartoCNN1//"
DIR_CKPTs = "CHECKPOINTS//BT_1//"
# DIR_CKPTs = "CHECKPOINTS//BT_0//"  # debug
CKPTs_CLASS = Quarto_bot
CKPTs_PARAMs = {"deterministic": False, "temperature": 0.1}

baselines = [
    {
        "path": "CHECKPOINTS//EXP_id03//20250922_1247-EXP_id03_epoch_0009.pt",
        "name": "bot_good",
        "bot": Quarto_bot,
        "params": {"deterministic": False, "temperature": 0.1},
    },
    {
        "path": "CHECKPOINTS//EXP_id03//20250922_1247-EXP_id03_epoch_0000.pt",
        "name": "bot_random",
        "bot": Quarto_bot,
        "params": {"deterministic": False, "temperature": 0.1},
    },
    {
        "path": "CHECKPOINTS//others//20250930_1010-EXP_id03_epoch_0017.pt",
        "name": "bot_Michael",
        "bot": Quarto_bot,
        "params": {"deterministic": False, "temperature": 0.1},
    },
]


## Loop in the checkpoint bot folder
checkpoint_files = glob.glob(os.path.join(DIR_CKPTs, "*.pt"))
print(checkpoint_files)
for bot_baseline in tqdm(baselines, desc="Loading baseline bots from folder"):
    baseline_name = bot_baseline["name"]
    # assumes all have the same class and options

    results = run_contest(
        player=CKPTs_CLASS(model_path=bot_baseline["path"], **bot_baseline["params"]),
        rivals=checkpoint_files,
        rival_class=CKPTs_CLASS,
        rival_options=CKPTs_PARAMs,
        matches=NUM_MATCHES,
        rivals_clip=-1,
        verbose=False,
        mode_2x2=True,
    )
    print(f"Results against {baseline_name}")
    pprint(results)
