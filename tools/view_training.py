# -*- coding: utf-8 -*-

"""
view_training - Plots the Bradley-Terry results of the players-epoch in the training.

Usage:
    view_training.py <tournament_file>
    view_training.py -h|--help
    view_training.py --version

Options:
    -h,--help               show help.
"""

"""
Python 3
06 / 06 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

from quartopy import logger
import numpy as np
import pandas as pd


# ----------------------------- #### --------------------------
from docopt import docopt

from collections import Counter

import logging
import os
import time

from datetime import datetime
from collections import Counter
from itertools import combinations, combinations_with_replacement, permutations


def bradley_terry_analysis(text_data, max_iters=1000, error_tol=1e-3):
    """Computes Bradley-Terry using iterative algorithm
    See: https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
    """
    # Do some aggregations for convenience
    # Total wins per excerpt
    winsA = text_data.groupby("Excerpt A").agg(sum)["Wins A"].reset_index()
    winsA = winsA[winsA["Wins A"] > 0]
    winsA.columns = ["Excerpt", "Wins"]
    winsB = text_data.groupby("Excerpt B").agg(sum)["Wins B"].reset_index()
    winsB = winsB[winsB["Wins B"] > 0]
    winsB.columns = ["Excerpt", "Wins"]
    wins = pd.concat([winsA, winsB]).groupby("Excerpt").agg(sum)["Wins"]

    # Total games played between pairs
    num_games = Counter()
    for index, row in text_data.iterrows():
        key = tuple(sorted([row["Excerpt A"], row["Excerpt B"]]))
        total = sum([row["Wins A"], row["Wins B"]])
        num_games[key] += total

    # Iteratively update 'ranks' scores
    excerpts = sorted(list(set(text_data["Excerpt A"]) | set(text_data["Excerpt B"])))
    ranks = pd.Series(np.ones(len(excerpts)) / len(excerpts), index=excerpts)
    for iters in range(max_iters):
        oldranks = ranks.copy()
        for excerpt in ranks.index:
            denom = np.sum(
                num_games[tuple(sorted([excerpt, p]))] / (ranks[p] + ranks[excerpt])
                for p in ranks.index
                if p != excerpt
            )
            ranks[excerpt] = 1.0 * wins[excerpt] / denom

        ranks /= sum(ranks)

        if np.sum((ranks - oldranks).abs()) < error_tol:
            break

    if np.sum((ranks - oldranks).abs()) < error_tol:
        logging.info(" * Converged after %d iterations.", iters)
    else:
        logging.info(" * Max iterations reached (%d iters).", max_iters)

    # Note we can control scaling here. For this competiton we have -'ve and positive values on the scale
    # To reproduce the results from example; I choose to multiply the rank with x100
    ranks = ranks.sort_values(ascending=False).apply(lambda x: x * 100).round(2)

    return ranks


# ####################################################################
def main(args):
    """

    ## Parameters

    ``a``:

    ``b``:

    ## Return

    ``a``:

    """
    import pickle

    pickle_file = args["<tournament_file>"]

    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    data
    n_epochs = len(data)
    total = 300

    df = pd.DataFrame()
    win_rate = np.full((n_epochs, n_epochs), np.nan)
    unilateral = np.full((n_epochs, n_epochs), np.nan)
    for epoch, results_epoch in enumerate(data):
        for rival_epoch, results_vs_rival in results_epoch.items():
            _in_favor = results_vs_rival["wins"] + (results_vs_rival["draws"]) * 0.5
            in_contra = results_vs_rival["losses"] + (results_vs_rival["draws"]) * 0.5

            win_rate[epoch, rival_epoch] = _in_favor
            unilateral[epoch, rival_epoch] = _in_favor
            total = _in_favor + in_contra
            win_rate[rival_epoch, epoch] = in_contra
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "Excerpt A": epoch,
                                "Excerpt B": rival_epoch,
                                "Wins A": _in_favor,
                                "Wins B": in_contra,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    scores = bradley_terry_analysis(df)
    print(scores)


if __name__ == "__main__":
    args = docopt(
        doc=__doc__,
        version="1",
    )
    logger.info(args)
    main(args)
