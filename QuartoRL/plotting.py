# -*- coding: utf-8 -*-

"""
Python 3
17 / 09 / 2025
@author: z_tjona


"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

"Either mathematics is too big for the human mind or the human mind is more than a machine."
-Kurt Godël
"""

import matplotlib.pyplot as plt
import numpy as np
from os import path
from datetime import datetime

plt.ion()  # Enable interactive mode


def plot_win_rate(
    *args: tuple[str | int, list[float]],
    SMOOTHING_WINDOW: int = 5,
    FREQ_EPOCH_SAVING: int = 1,
    FOLDER_SAVE: str = "./",
    FIG_NAME=lambda epoch: f"{datetime.now().strftime('%Y%m%d_%H%M')}-win_rate_{epoch:04d}.svg",
):
    """Plot win rate over epochs for multiple rivals.
    Args:
        *args: Tuples of rival name and their corresponding win rates list.
        SMOOTHING_WINDOW (int): Window size for moving average smoothing. If -1, no smoothing is applied.
    """
    plt.figure(1, figsize=(8, 6), clear=True)
    for rival_name, win_rates in args:
        plt.scatter(
            np.arange(len(win_rates)),
            win_rates,
            marker="o",  # type: ignore
            s=10,
            linestyle="",
            alpha=0.3,
            label=f"vs {rival_name}",
        )
        if SMOOTHING_WINDOW > 1 and len(win_rates) >= SMOOTHING_WINDOW:
            smoothed = np.convolve(
                win_rates,
                np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW,
                mode="valid",
            )
            offset = SMOOTHING_WINDOW - 1
            plt.plot(
                np.arange(offset // 2, len(smoothed) + offset // 2),
                smoothed,
                alpha=0.6,
            )

    plt.xlabel("Training epochs")
    plt.ylabel("Win rate")
    plt.title(f"Win rate of the epoch player vs rivals")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

    # Save the figure at regular intervals
    if len(win_rates) % FREQ_EPOCH_SAVING == 0 and FREQ_EPOCH_SAVING != -1:
        plt.savefig(
            path.join(FOLDER_SAVE, FIG_NAME(len(win_rates))),
            dpi=300,
            bbox_inches="tight",
        )


def plot_loss(
    loss_data: dict[str, list[float | int]],
    FREQ_EPOCH_SAVING: int = 200,
    FOLDER_SAVE: str = "./",
    FIG_NAME=lambda epoch: f"{datetime.now().strftime('%Y%m%d_%H%M')}-loss_{epoch:04d}.svg",
    SHOW_EPOCH_LINES: int = 500,
):
    """
    FREQ_EPOCH_SAVING: int if -1 no saving, else save every n epochs
    SHOW_EPOCH_LINES: int if >= N_ITERS show epoch lines
    """
    epoch_values = loss_data["epoch_values"]
    loss_values = loss_data["loss_values"]
    N_ITERS = len(loss_values)
    # Plot loss values
    plt.figure(2, figsize=(10, 5), clear=True)
    plt.plot(np.arange(N_ITERS), loss_values, ".", alpha=0.6, label="Loss values")
    if SHOW_EPOCH_LINES >= N_ITERS:
        # Add vertical lines at each epoch value
        for epoch in epoch_values:
            plt.axvline(x=epoch, color="r", linestyle="--", alpha=0.3)
        plt.plot([], [], "r--", alpha=0.3, label="Epoch boundaries")

    plt.xlabel("Training iterations")
    plt.ylabel("Loss")
    plt.title(f"Training Loss up to epoch {len(epoch_values)}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

    # Save the figure at regular intervals
    if len(epoch_values) % FREQ_EPOCH_SAVING == 0 and FREQ_EPOCH_SAVING != -1:
        plt.savefig(
            path.join(FOLDER_SAVE, FIG_NAME(len(epoch_values))),
            dpi=300,
            bbox_inches="tight",
        )


def plot_contest_results(epochs_results: list[dict[int, dict[str, int]]]):
    """[LEGACY] Calculate win rate by rival and plot it."""

    _n_epochs = len(epochs_results)
    _n_rivals = _n_epochs
    # Assuming rivals are from previous epochs

    # Win rate by epoch and rival
    win_rate = np.full((_n_epochs, _n_rivals), np.nan)

    for player_id, player_results in enumerate(epochs_results):
        for rival_id, result_vs_rival in player_results.items():
            _total = (
                result_vs_rival["wins"]
                + result_vs_rival["draws"]
                + result_vs_rival["losses"]
            )
            _w_rate = (
                result_vs_rival["wins"] + 0.5 * result_vs_rival["draws"]
            ) / _total

            win_rate[player_id, rival_id] = _w_rate

    plt.figure(1, figsize=(8, 6), clear=True)
    im = plt.imshow(win_rate, aspect="auto", interpolation="none", cmap="viridis")
    plt.colorbar(im, label="Win Rate", extend="neither")
    im.set_clim(0, 1)
    plt.xlabel("Rival")
    plt.ylabel("Epoch")
    # plt.xticks(ticks=range(_n_rivals))
    # plt.yticks(ticks=range(_n_epochs))
    plt.title("Win Rate by Epoch vs Rival")
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
