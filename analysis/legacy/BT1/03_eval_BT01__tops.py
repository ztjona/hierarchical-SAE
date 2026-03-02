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

from bot.CNN_bot import Quarto_bot
from bot.CNN_F_bot import Quarto_bot as F_bot
from QuartoRL import run_contest

from tqdm.auto import tqdm
from pprint import pprint

import os
import glob
import pickle
import datetime

checkpoint_idxs = [
    1153,
    889,
    4321,
    1572,
    7003,
    719,
    4432,
    5079,
    7169,
    7438,
    2910,
    1096,
    855,
    852,
    1754,
    726,
    1884,
    6004,
    851,
    5296,
    2135,
    1470,
    3730,
    3815,
    2965,
    7304,
    773,
    1245,
    16,
    4383,
]

RESULTS_FILE = "eval_BT01__tops_results.pkl"
NUM_MATCHES = 30
DIR_CKPTs = "CHECKPOINTS//BT_1//"
# DIR_CKPTs = "CHECKPOINTS//BT_0//"  # debug
CKPTs_CLASS = Quarto_bot
CKPTs_PARAMs = {"deterministic": False, "temperature": 0.1}

# get path
checkpoint_files = []
for idx in checkpoint_idxs:
    _p = glob.glob(os.path.join(DIR_CKPTs, f"*_{idx:04d}.pt"))
    assert len(_p) == 1, f"Checkpoint for idx {idx} not found or multiple found {_p}."
    checkpoint_files.append(_p[0])

results = {}
for idx, ckp in tqdm(
    zip(checkpoint_idxs, checkpoint_files), desc="Evaluating checkpoints in tournament"
):
    print(f"Evaluating checkpoint {idx} at {ckp}")
    _results = run_contest(
        player=CKPTs_CLASS(model_path=ckp, **CKPTs_PARAMs),
        rivals=checkpoint_files,
        rival_class=CKPTs_CLASS,
        rival_names=checkpoint_idxs,  # type: ignore
        rival_options=CKPTs_PARAMs,
        matches=NUM_MATCHES,
        rivals_clip=-1,
        verbose=False,
        mode_2x2=True,
    )
    print(f"Results of {idx}")
    pprint(_results)
    results[idx] = dict(_results)

with open(RESULTS_FILE, "wb") as f:
    pickle.dump(results, f)
