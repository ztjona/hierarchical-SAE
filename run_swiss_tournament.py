# -*- coding: utf-8 -*-

"""
Python 3
28 / 10 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

"Either mathematics is too big for the human mind or the human mind is more than a machine."
-Kurt Godël
"""


# ----------------------------- MAIN CONFIGS --------------------------
from bot.CNN_bot import Quarto_bot


FOLDER_CHECKPOINTS = "CHECKPOINTS//E02_win_rate//"
BOTs_CLASS = Quarto_bot
BOTs_PARAMs = {"deterministic": False, "temperature": 0.1}
NUM_ROUNDS = 5
DOUBLE_SWISS = True  # Each bot plays once as player 1 and once as player 2
MODE_McMahon = True  # Initial points based on estimated strength "Epoch"


# ----------------------------- logging config --------------------------
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

def run_swiss_tournament():
