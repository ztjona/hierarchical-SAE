# -*- coding: utf-8 -*-

"""
Python 3
04 / 12 / 2025
@author: z_tjona


"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

"Either mathematics is too big for the human mind or the human mind is more than a machine."
-Kurt Godël
"""


from quartopy import Board
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from datetime import datetime
from os import path


def plot_boards_comp(
    *boards_pair: tuple[Board, Board],
    q_place: torch.Tensor,
    q_select: torch.Tensor,
    fig_num: int = 3,
    DISPLAY_PLOT: bool = True,
    MAX_BOARDS: int = 6,
    position: tuple[int, int] | None = (500, 0),
    experiment_name: str = "",
    FREQ_EPOCH_SAVING: int = -1,
    FOLDER_SAVE: str = "./",
    FIG_NAME=lambda epoch: f"{datetime.now().strftime('%Y%m%d_%H%M')}-boards_comp_{epoch:04d}.svg",
    current_epoch: int = 0,
) -> None:
    """Plot pairs of boards side by side in a 2xn subplot grid (transposed).

    Parameters
    ----------
    *boards_pair : tuple[Board, Board]
        Variable number of board pairs to compare. Typically (state, next_state).
    fig_num : int
        Figure number to use for plotting (default: 3)
    DISPLAY_PLOT : bool
        Whether to display the plot interactively (default: True)
    MAX_BOARDS : int
        Maximum number of board pairs to display. If more pairs provided,
        randomly samples MAX_BOARDS pairs (default: 6)
    position : tuple[int, int], optional
        (x, y) position in pixels for top-left corner of figure window
    experiment_name : str
        Experiment name to include in figure window title (default: "")
    FREQ_EPOCH_SAVING : int
        If -1, no saving. Otherwise, save figure every n epochs (default: -1)
    FOLDER_SAVE : str
        Directory path to save figures (default: "./")
    FIG_NAME : callable
        Lambda function that generates filename given epoch number
    current_epoch : int
        Current epoch number for saving (default: 0)
    """
    n = len(boards_pair)
    if n == 0:
        return

    # Limit to MAX_BOARDS random samples
    if n > MAX_BOARDS:
        indices = random.sample(range(n), MAX_BOARDS)
        boards_pair = tuple(boards_pair[i] for i in sorted(indices))
        n = MAX_BOARDS

    # Create 2xn subplot grid (transposed) with adaptive sizing
    # Retrieve existing figure or create new one
    experiment_name = f"{experiment_name}-{fig_num}"
    if plt.fignum_exists(experiment_name):
        fig = plt.figure(experiment_name)
        fig.clf()  # Clear figure content but keep the window
    else:
        fig = plt.figure(experiment_name, figsize=(16, 9), constrained_layout=True)

    # Set window position if specified
    if position is not None:
        try:
            manager = fig.canvas.manager  # type: ignore
            manager.window.wm_geometry(f"+{position[0]}+{position[1]}")  # type: ignore
        except:
            pass  # Silently fail if backend doesn't support positioning

    axes = fig.subplots(2, n)

    # Handle single pair case (axes won't be 2D)
    if n == 1:
        axes = np.array(axes).reshape(-1, 1)

    # Plot each pair (transposed: rows are board states, columns are pairs)
    for i, (b1, b2) in enumerate(boards_pair):
        b1.plot(title=b1.name, ax=axes[0, i], show=False)  # type: ignore
        b2.plot(title=b2.name, ax=axes[1, i], show=False)  # type: ignore

    # Save the figure at regular intervals
    if current_epoch % FREQ_EPOCH_SAVING == 0 and FREQ_EPOCH_SAVING != -1:
        plt.savefig(
            path.join(FOLDER_SAVE, FIG_NAME(current_epoch)),
            dpi=1000,
            bbox_inches="tight",
        )

    if DISPLAY_PLOT:
        plt.draw()
        plt.pause(0.001)


def plot_Qv_progress(
    q_values_history: dict[str, list[torch.Tensor]],
    rewards: torch.Tensor,
    fig_num: int = 4,
    DISPLAY_PLOT: bool = True,
    done_v: torch.Tensor | None = None,
    PLOT_TYPE: str = "time_series",
    position: tuple[int, int] | None = (0, 0),
    experiment_name: str = "",
    FREQ_EPOCH_SAVING: int = -1,
    FOLDER_SAVE: str = "./",
    FIG_NAME=lambda epoch: f"{datetime.now().strftime('%Y%m%d_%H%M')}-qv_progress_{epoch:04d}.svg",
    current_epoch: int = 0,
) -> None:
    """Plot Q-value progression over epochs for each sample in the batch.

    Parameters
    ----------
    q_values_history : dict[str, list[torch.Tensor]]
        Dictionary with keys 'q_place' and 'q_select', each containing a list of
        tensors (one per epoch) with Q-values for each sample
    rewards : torch.Tensor
        Target rewards for each sample (batch_size,)
    fig_num : int
        Figure number to use for plotting (default: 4)
    DISPLAY_PLOT : bool
        Whether to display the plot interactively (default: True)
    done_v : torch.Tensor, optional
        Boolean tensor indicating whether each sample is a terminal state (batch_size,).
        Terminal states are plotted with higher prominence (thicker, more opaque lines).
    position : tuple[int, int], optional
        (x, y) position in pixels for top-left corner of figure window
    experiment_name : str
        Experiment name to include in figure window title (default: "")
    FREQ_EPOCH_SAVING : int
        If -1, no saving. Otherwise, save figure every n epochs (default: -1)
    FOLDER_SAVE : str
        Directory path to save figures (default: "./")
    FIG_NAME : callable
        Lambda function that generates filename given epoch number
    current_epoch : int
        Current epoch number for saving (default: 0)
    """
    if not q_values_history or len(q_values_history.get("q_place", [])) == 0:
        return

    # Extract Q-values
    q_place_history = q_values_history.get("q_place", [])
    q_select_history = q_values_history.get("q_select", [])

    batch_size = q_place_history[0].shape[0] if q_place_history else 0
    n_epochs = len(q_place_history)

    if batch_size == 0:
        return

    epochs = np.arange(n_epochs)

    # Retrieve existing figure or create new one
    experiment_name = f"{experiment_name}-{fig_num}"
    if plt.fignum_exists(experiment_name):
        fig = plt.figure(experiment_name)
        fig.clf()
    else:
        fig = None  # Will be created below with appropriate size

    if fig is None:
        fig = plt.figure(experiment_name, figsize=(16, 10), constrained_layout=True)

    # Set window position if specified
    if position is not None:
        try:
            manager = fig.canvas.manager  # type: ignore
            manager.window.wm_geometry(f"+{position[0]}+{position[1]}")  # type: ignore
        except:
            pass

    axes = fig.subplots(2, 3)

    # Split samples by reward value (round to handle decimal rewards)
    loss_indices = [i for i in range(batch_size) if round(rewards[i].item()) == -1]
    draw_indices = [i for i in range(batch_size) if round(rewards[i].item()) == 0]
    win_indices = [i for i in range(batch_size) if round(rewards[i].item()) == 1]

    # Define plot configurations: (row, col, indices, q_history, title)
    plot_configs = [
        (0, 0, loss_indices, q_place_history, "Q_place: R=-1"),
        (0, 1, draw_indices, q_place_history, "Q_place: R=0"),
        (0, 2, win_indices, q_place_history, "Q_place: R=1"),
        (1, 0, loss_indices, q_select_history, "Q_select: R=-1"),
        (1, 1, draw_indices, q_select_history, "Q_select: R=0"),
        (1, 2, win_indices, q_select_history, "Q_select: R=1"),
    ]

    if PLOT_TYPE == "time_series":
        # plot 6 aggregated curves grouped by reward value

        for row, col, indices, q_history, title in plot_configs:
            ax = axes[row, col]  # type: ignore

            # Determine target reward from title
            target_reward = -1 if "R=-1" in title else (0 if "R=0" in title else 1)

            # Plot individual Q-value trajectories
            q_values_all = []  # Collect all Q-values for computing mean
            for i in indices:
                q_sample = [q[i].item() for q in q_history]
                q_values_all.append(q_sample)
                is_terminal = done_v[i].item() if done_v is not None else False
                ax.plot(
                    epochs,
                    q_sample,
                    "-",
                    alpha=0.2 if is_terminal else 0.1,
                    linewidth=1.0 if is_terminal else 0.5,
                    color="gray",
                )

            # Add reference line at target reward (expected convergence)
            ax.axhline(
                y=target_reward,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"Target={target_reward}",
            )

            # Plot mean Q-value trajectory with confidence interval
            if q_values_all:
                q_array = np.array(q_values_all)  # shape: (n_samples, n_epochs)
                q_mean = np.mean(q_array, axis=0)
                q_std = np.std(q_array, axis=0)

                # Mean line
                ax.plot(
                    epochs,
                    q_mean,
                    "b-",
                    linewidth=3,
                    alpha=0.9,
                    label=f"Mean Q",
                    zorder=10,
                )

                # Confidence interval (±1 std)
                ax.fill_between(
                    epochs,
                    q_mean - q_std,
                    q_mean + q_std,
                    alpha=0.2,
                    color="blue",
                    label="±1 std",
                )

                # Show final convergence error in title
                final_error = abs(q_mean[-1] - target_reward)
                ax.set_title(f"{title}\nFinal Error: {final_error:.3f}")
            else:
                ax.set_title(title)

            # Only show x-label on bottom row
            if row == 1:
                ax.set_xlabel("Epoch")
            # Only show y-label on leftmost column
            if col == 0:
                ax.set_ylabel("Q-value")

            ax.set_ylim(-1.2, 1.2)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

    elif PLOT_TYPE == "hist":
        # Create histogram evolution plots showing Q-value distribution over epochs
        HIST_BINS = 50
        HIST_RANGE = (-1.1, 1.1)

        for row, col, indices, q_history, title in plot_configs:
            ax = axes[row, col]  # type: ignore

            if not indices:  # Skip if no samples in this group
                ax.text(
                    0.5,
                    0.5,
                    "No samples",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(title)
                continue

            # Compute histograms for this reward group across epochs
            hist_data = []
            for q_epoch in q_history:
                q_subset = q_epoch[indices].detach().cpu().numpy().flatten()
                hist, _ = np.histogram(q_subset, bins=HIST_BINS, range=HIST_RANGE)
                # Normalize to percentage
                hist_percent = (hist / hist.sum()) * 100 if hist.sum() > 0 else hist
                hist_data.append(hist_percent)

            # Plot histogram evolution
            if hist_data:
                hist_array = np.array(hist_data)
                im = ax.imshow(
                    hist_array.T,
                    aspect="auto",
                    origin="lower",
                    cmap="viridis",
                    interpolation="nearest",
                    extent=[0, n_epochs, HIST_RANGE[0], HIST_RANGE[1]],
                )
                # Only show x-label on bottom row
                if row == 1:
                    ax.set_xlabel("Epoch")
                # Only show y-label on leftmost column
                if col == 0:
                    ax.set_ylabel("Q-value")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                plt.colorbar(im, ax=ax, label="Percentage (%)")

    # Save the figure at regular intervals
    if current_epoch % FREQ_EPOCH_SAVING == 0 and FREQ_EPOCH_SAVING != -1:
        plt.savefig(
            path.join(FOLDER_SAVE, FIG_NAME(current_epoch)),
            dpi=1000,
            bbox_inches="tight",
        )

    if DISPLAY_PLOT:
        plt.draw()
        plt.pause(0.001)
