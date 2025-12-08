# -*- coding: utf-8 -*-


"""
Python 3
26 / 05 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""
# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime
from colorama import Fore, Style, Back, init as colorama_init


colorama_init(autoreset=True)


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"


handler = logging.StreamHandler(stdout)
handler.setLevel(logging.DEBUG)
formatter = ColorFormatter(
    f"{Fore.GREEN + Back.RED}[%(asctime)s]{Back.RESET}[%(levelname)s]{Back.WHITE}[%(filename)s]{Back.RESET} %(message)s",
    datefmt="%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)

logger = logging.getLogger("TrainRL")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False  # Prevent duplicate logs from root logger
logger.debug("Creating logger for TrainRL")
logger.debug(datetime.now())
