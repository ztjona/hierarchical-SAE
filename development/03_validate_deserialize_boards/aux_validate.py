# -*- coding: utf-8 -*-

"""
Python 3
02 / 12 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

"Either mathematics is too big for the human mind or the human mind is more than a machine."
-Kurt Godël
"""

# ----------------------------- logging --------------------------
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

# exp.board_state[13]
board_serial = "0000000010000000001000000000000000000000000010000000000001000000000000000000000000000100000000000000000000000000000000000000001000010000000000000000000000000000100000000000000000000000000000000000001000000000000000000010000000000001000000000100000000000000"

from quartopy import Board

board = Board.serialized_2_board(board_serial)
print(board)
serialized = Board.serialize(board)
board2 = Board.serialized_2_board(serialized)
print(board2)
fig = board2.plot("asfdb")
fig.show()  # or display it in a notebook, save it, etc.
print(board2 == board)
