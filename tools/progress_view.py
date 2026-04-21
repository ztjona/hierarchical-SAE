# -*- coding: utf-8 -*-

"""
progress_view - Counts the number of checkpoints for a given experiment

Usage:
    progress_view.py <EXPERIMENT_NAME>
    progress_view.py -h|--help
    progress_view.py --version

Options:
    -h,--help               show help.
"""

"""
Python 3
20 / 04 / 2026
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

"Either mathematics is too big for the human mind or the human mind is more than a machine."
-Kurt Godël
"""
from pathlib import Path

checkpoint_folder = ".//CHECKPOINTS//"
CHECK_EXTENSION = ".pt"

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


# ----------------------------- #### --------------------------
from docopt import docopt


# ####################################################################
def main(args):
    check_pattern = f'{args["<EXPERIMENT_NAME>"].lower()}('

    for f in Path(checkpoint_folder).iterdir():
        if f.is_dir() and check_pattern in f.name.lower():
            logging.debug(f"Found checkpoint file: {f.name}")
            count = len(list(f.absolute().glob(f"*{CHECK_EXTENSION}")))
            logging.info(f"Experiment '{f.name}': {count} checkpoints found.")

    return


if __name__ == "__main__":
    args = docopt(
        doc=__doc__,
        version="1",
    )
    logging.debug(args)
    main(args)
