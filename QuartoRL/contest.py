# -*- coding: utf-8 -*-

"""
Python 3
04 / 06 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""
from quartopy import BotAI, play_games

from collections import defaultdict
from utils.logger import logger
import random
from tqdm import tqdm
from os import path


def contest_2_win_rate(
    contest_results: dict[str | int, dict[str, int]],
) -> dict[str | int, float]:
    """Convert contest results to win rates.
    Args:
        contest_results (dict): Contest results vs rival with wins, losses, and draws.
    Returns:
        dict: Win rates vs each rival.
    """
    win_rates: dict[str | int, float] = {}
    for rival_name, results in contest_results.items():
        total_games = results["wins"] + results["losses"] + results["draws"]

        win_rates[rival_name] = (results["wins"] + results["draws"] * 0.5) / total_games
    return win_rates


# ####################################################################
def run_contest(
    player: BotAI,
    rivals: list[str],
    rival_class: type[BotAI] | list[type[BotAI]],
    rival_names: list[str | int] = [],
    rival_options: dict | list[dict] = {},
    matches: int = 100,
    rivals_clip: int = -1,
    verbose: bool = True,
    mode_2x2: bool = False,
    PROGRESS_MESSAGE: str = "Playing tournament matches...",
):
    """Run a contest between a player and multiple rivals.
    Args:
        player (BotAI): The player bot.
        rivals (list[str]): List of file paths to rival bots.
        rival_class (type[BotAI] | list[type[BotAI]]): Class type of the rival bots.
            If a single type, all rivals use the same class.
            If a list, must match the length of rivals list.
        rival_names (list[str | int]): List of names for the rivals.
        rival_options (dict | list[dict]): Options to pass to rival bot constructors.
            If a single dict, all rivals use the same options.
            If a list, must match the length of rivals list.
        matches (int): Total number of matches to play against each rival.
        rivals_clip (int): Limit the number of rivals to consider. -1 means no limit.
        verbose (bool): Whether to print detailed logs.
        mode_2x2 (bool): Whether to use 2x2 victory mode.
        PROGRESS_MESSAGE (str): Message to display in progress bar.
    """
    n_rivals = len(rivals)
    logger.debug(f"Running contest with {n_rivals} rivals, {matches} matches")

    # Normalize rival_class and rival_options to lists
    if not isinstance(rival_class, list):
        rival_class_list = [rival_class] * n_rivals
    else:
        rival_class_list = rival_class
        assert (
            len(rival_class_list) == n_rivals
        ), f"rival_class list length ({len(rival_class_list)}) must match rivals length ({n_rivals})"

    if not isinstance(rival_options, list):
        rival_options_list = [rival_options] * n_rivals
    else:
        rival_options_list = rival_options
        assert (
            len(rival_options_list) == n_rivals
        ), f"rival_options list length ({len(rival_options_list)}) must match rivals length ({n_rivals})"

    selected = range(n_rivals)  # Default to all rivals
    if rivals_clip == -1:
        logger.debug("No clipping of rivals, using all available rivals")
    elif rivals_clip > n_rivals:
        logger.warning(
            f"Cannot clip to requested {rivals_clip}. Playing against all {n_rivals} rivals"
        )

    else:
        logger.debug(f"Clipping rivals to {rivals_clip} random rivals")
        selected = sorted(random.sample(range(n_rivals), k=rivals_clip))

    if not rival_names:
        rival_names = [path.basename(r) for r in rivals]
    rivals_selected = {rival_names[i]: (rivals[i], i) for i in selected}

    # index del rival: {"wins": 0, "losses": 0, "draws": 0}
    results: dict[int | str, dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "draws": 0}
    )
    for idx, (rival_name, (rival_file, rival_idx)) in enumerate(
        tqdm(rivals_selected.items(), desc=PROGRESS_MESSAGE)
    ):
        rival = rival_class_list[rival_idx](
            model_path=rival_file, **rival_options_list[rival_idx]
        )

        logger.debug(f"Playing against rival {rival_name}/{len(rivals)}: {rival.name}")
        _, win_rate_p1 = play_games(
            matches=matches // 2,
            player1=player,
            player2=rival,
            verbose=verbose,
            save_match=False,
            mode_2x2=mode_2x2,
            PROGRESS_MESSAGE="",
        )
        logger.debug(win_rate_p1)
        results[rival_name]["wins"] += win_rate_p1["Player 1"]
        results[rival_name]["losses"] += win_rate_p1["Player 2"]
        results[rival_name]["draws"] += win_rate_p1["Tie"]

        _, win_rate_p2 = play_games(
            matches=matches // 2,
            player1=rival,
            player2=player,
            verbose=verbose,
            save_match=False,
            mode_2x2=mode_2x2,
            PROGRESS_MESSAGE=PROGRESS_MESSAGE,
        )
        logger.debug(win_rate_p2)
        results[rival_name]["wins"] += win_rate_p2["Player 2"]
        results[rival_name]["losses"] += win_rate_p2["Player 1"]
        results[rival_name]["draws"] += win_rate_p2["Tie"]

    return results
