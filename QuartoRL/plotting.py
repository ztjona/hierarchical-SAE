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
    DISPLAY_PLOT: bool = False,
    fig_num: int = 1,
    position: tuple[int, int] | None = (500, 600),
    experiment_name: str = "",
):
    """Plot win rate over epochs for multiple rivals.

    Parameters
    ----------
    *args : tuple[str | int, list[float]]
        Variable number of (rival_name, win_rates) tuples to plot
    SMOOTHING_WINDOW : int
        Window size for moving average smoothing. If -1 or <=1, no smoothing is applied (default: 5)
    FREQ_EPOCH_SAVING : int
        If -1, no saving. Otherwise, save figure every n epochs (default: 1)
    FOLDER_SAVE : str
        Directory path to save figures (default: "./")
    FIG_NAME : callable
        Lambda function that generates filename given epoch number
    DISPLAY_PLOT : bool
        Whether to display the plot interactively (default: False)
    fig_num : int
        Figure number to use for plotting (default: 1)
    position : tuple[int, int], optional
        (x, y) position in pixels for top-left corner of figure window
    experiment_name : str
        Experiment name to include in figure window title (default: "")
    """
    if not DISPLAY_PLOT:
        plt.ioff()  # Disable interactive mode

    # Retrieve existing figure or create new one
    experiment_name = f"{experiment_name}-{fig_num}"
    if plt.fignum_exists(experiment_name):
        fig = plt.figure(experiment_name)
        fig.clf()  # Clear figure content but keep the window
    else:
        fig = plt.figure(experiment_name, figsize=(8, 6))

    # Set window position if specified
    if position is not None:
        try:
            manager = fig.canvas.manager  # type: ignore
            manager.window.wm_geometry(f"+{position[0]}+{position[1]}")  # type: ignore
        except:
            pass

    for rival_name, win_rates in args:
        epochs_arr = np.arange(len(win_rates))
        win_rates_arr = np.array(win_rates)

        # Scatter plot of raw data
        plt.scatter(
            epochs_arr,
            win_rates_arr,
            marker="o",  # type: ignore
            s=10,
            linestyle="",
            alpha=0.3,
            label=f"vs {rival_name}",
        )

        if SMOOTHING_WINDOW > 1 and len(win_rates) >= SMOOTHING_WINDOW:
            # Calculate smoothed mean
            smoothed = np.convolve(
                win_rates_arr,
                np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW,
                mode="valid",
            )
            offset = SMOOTHING_WINDOW - 1
            smoothed_epochs = np.arange(offset // 2, len(smoothed) + offset // 2)

            # Calculate rolling std for error band
            window_stds = []
            for i in range(len(smoothed)):
                window_data = win_rates_arr[i : i + SMOOTHING_WINDOW]
                window_stds.append(np.std(window_data))
            window_stds = np.array(window_stds)

            # Plot smoothed line
            line = plt.plot(
                smoothed_epochs,
                smoothed,
                alpha=0.8,
                linewidth=2,
            )[0]

            # Add std error band with same color as line
            plt.fill_between(
                smoothed_epochs,
                smoothed - window_stds,  # type: ignore
                smoothed + window_stds,  # type: ignore
                alpha=0.2,
                color=line.get_color(),
            )

    plt.xlabel("Training epochs")
    plt.ylabel("Win rate")
    plt.title(f"Win rate of the epoch player vs rivals")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if DISPLAY_PLOT:
        plt.draw()
        plt.pause(0.001)

    # Save the figure at regular intervals
    if len(win_rates) % FREQ_EPOCH_SAVING == 0 and FREQ_EPOCH_SAVING != -1:
        plt.savefig(
            path.join(FOLDER_SAVE, FIG_NAME(len(win_rates))),
            dpi=1000,
            bbox_inches="tight",
        )


def plot_loss(
    loss_data: dict[str, list[float | int]],
    FREQ_EPOCH_SAVING: int = 200,
    FOLDER_SAVE: str = "./",
    FIG_NAME=lambda epoch: f"{datetime.now().strftime('%Y%m%d_%H%M')}-loss_{epoch:04d}.svg",
    DISPLAY_PLOT: bool = False,
    fig_num: int = 2,
    position: tuple[int, int] | None = (0, 600),
    experiment_name: str = "",
):
    """
    Plot average loss per epoch with standard deviation error bands.

    Parameters
    ----------
    loss_data : dict[str, list[float | int]]
        Dictionary with 'loss_values' (list of all iteration losses) and
        'epoch_values' (list of iteration indices marking epoch boundaries)
    FREQ_EPOCH_SAVING : int
        If -1, no saving. Otherwise, save figure every n epochs (default: 200)
    FOLDER_SAVE : str
        Directory path to save figures (default: "./")
    FIG_NAME : callable
        Lambda function that generates filename given epoch number
    DISPLAY_PLOT : bool
        Whether to display the plot interactively (default: False)
    fig_num : int
        Figure number to use for plotting (default: 2)
    position : tuple[int, int], optional
        (x, y) position in pixels for top-left corner of figure window
    experiment_name : str
        Experiment name to include in figure window title (default: "")
    """
    if not DISPLAY_PLOT:
        plt.ioff()  # Disable interactive mode
    epoch_values = loss_data["epoch_values"]
    loss_values = loss_data["loss_values"]

    # Retrieve existing figure or create new one
    experiment_name = f"{experiment_name}-{fig_num}"
    if plt.fignum_exists(experiment_name):
        fig = plt.figure(experiment_name)
        fig.clf()  # Clear figure content but keep the window
    else:
        fig = plt.figure(experiment_name, figsize=(10, 5))

    # Set window position if specified
    if position is not None:
        try:
            manager = fig.canvas.manager  # type: ignore
            manager.window.wm_geometry(f"+{position[0]}+{position[1]}")  # type: ignore
        except:
            pass

    # Calculate mean and std for each epoch
    n_epochs = len(epoch_values)
    epoch_means = []
    epoch_stds = []

    for i in range(n_epochs):
        # Get start and end indices for this epoch
        start_idx = epoch_values[i]
        end_idx = epoch_values[i + 1] if i + 1 < n_epochs else len(loss_values)

        # Extract losses for this epoch
        epoch_losses = loss_values[start_idx:end_idx]

        if len(epoch_losses) > 0:
            epoch_means.append(np.mean(epoch_losses))
            epoch_stds.append(np.std(epoch_losses))
        else:
            epoch_means.append(np.nan)
            epoch_stds.append(np.nan)

    epoch_means = np.array(epoch_means)
    epoch_stds = np.array(epoch_stds)
    epochs = np.arange(n_epochs)

    # Plot mean loss line with std error band
    line = plt.plot(
        epochs, epoch_means, ".-", alpha=0.8, linewidth=2, label="Mean loss"
    )[0]
    plt.fill_between(
        epochs,
        epoch_means - epoch_stds,  # type: ignore
        epoch_means + epoch_stds,  # type: ignore
        alpha=0.3,
        color=line.get_color(),
        label="±1 std dev",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss up to epoch {n_epochs}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if DISPLAY_PLOT:
        plt.draw()
        plt.pause(0.001)

    # Save the figure at regular intervals
    if n_epochs % FREQ_EPOCH_SAVING == 0 and FREQ_EPOCH_SAVING != -1:
        plt.savefig(
            path.join(FOLDER_SAVE, FIG_NAME(n_epochs)),
            dpi=1000,
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

    # Retrieve existing figure or create new one
    if plt.fignum_exists(1):
        fig = plt.figure(1)
        fig.clf()
    else:
        fig = plt.figure(1, figsize=(8, 6))

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
