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
    SAVEFIG_DPI: int = 1000,
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
            dpi=SAVEFIG_DPI,
            bbox_inches="tight",
        )

    if DISPLAY_PLOT:
        plt.draw()
        plt.pause(0.001)


def plot_Qv_progress(
    q_values_history: dict[str, list[torch.Tensor]],
    group_values: torch.Tensor,
    fig_num: int = 4,
    DISPLAY_PLOT: bool = True,
    done_v: torch.Tensor | None = None,
    PLOT_TYPE: str = "time_series",
    group_label: str = "Outcome",
    position: tuple[int, int] | None = (0, 0),
    experiment_name: str = "",
    middle_column_title: str = "",
    FREQ_EPOCH_SAVING: int = -1,
    FOLDER_SAVE: str = "./",
    FIG_NAME=lambda epoch: f"{datetime.now().strftime('%Y%m%d_%H%M')}-qv_progress_{epoch:04d}.svg",
    current_epoch: int = 0,
    SAVEFIG_DPI: int = 1000,
) -> None:
    """Plot Q-value progression over epochs for each sample in the batch.

    Parameters
    ----------
    q_values_history : dict[str, list[torch.Tensor]]
        Dictionary with keys 'q_place' and 'q_select', each containing a list of
        tensors (one per epoch) with Q-values for each sample
    group_values : torch.Tensor
        Perspective labels used to group samples (batch_size,)
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
    middle_column_title : str
        Optional title prefix shown in middle-column subplots (default: "")
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

    # Extract Q-values, normalizing to numpy arrays (values may have been stored as plain lists)
    q_place_history = [
        np.array(q) if isinstance(q, list) else q
        for q in q_values_history.get("q_place", [])
    ]
    q_select_history = [
        np.array(q) if isinstance(q, list) else q
        for q in q_values_history.get("q_select", [])
    ]

    # Normalize grouping labels to numpy array (may be tensor, list, or array)
    if isinstance(group_values, list):
        group_values = np.array(group_values)
    elif hasattr(group_values, "detach"):
        group_values = group_values.detach().cpu().numpy()

    # Use minimum size across all epochs to ensure all indices are valid
    # This handles cases where different epochs have different numbers of samples
    if not q_place_history:
        return

    min_size_across_epochs = min(q.shape[0] for q in q_place_history)
    batch_size = min(group_values.shape[0], min_size_across_epochs)
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

    # Split samples by perspective group value.
    rounded_groups = np.rint(group_values[:batch_size]).astype(int)
    loss_indices = [i for i in range(batch_size) if rounded_groups[i] == -1]
    draw_indices = [i for i in range(batch_size) if rounded_groups[i] == 0]
    win_indices = [i for i in range(batch_size) if rounded_groups[i] == 1]

    # Define plot configurations: (row, col, indices, q_history, head name, ref value)
    plot_configs = [
        (0, 0, loss_indices, q_place_history, "Q_place", -1),
        (0, 1, draw_indices, q_place_history, "Q_place", 0),
        (0, 2, win_indices, q_place_history, "Q_place", 1),
        (1, 0, loss_indices, q_select_history, "Q_select", -1),
        (1, 1, draw_indices, q_select_history, "Q_select", 0),
        (1, 2, win_indices, q_select_history, "Q_select", 1),
    ]

    if PLOT_TYPE == "time_series":
        # plot 6 aggregated curves grouped by perspective value

        for row, col, indices, q_history, head_name, reference_value in plot_configs:
            ax = axes[row, col]  # type: ignore
            title = f"{head_name}: {group_label}={reference_value}"

            # Plot individual Q-value trajectories
            q_values_all = []  # Collect all Q-values for computing mean
            for i in indices:
                q_sample = np.array([q[i].item() for q in q_history], dtype=float)
                if not np.isfinite(q_sample).any():
                    continue
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

            # Add reference line at the grouping value for orientation.
            ax.axhline(
                y=reference_value,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"Reference={reference_value}",
            )

            # Plot mean Q-value trajectory with confidence interval
            if q_values_all:
                q_array = np.array(q_values_all)  # shape: (n_samples, n_epochs)
                q_mean = np.nanmean(q_array, axis=0)
                q_std = np.nanstd(q_array, axis=0)
                finite_mean_mask = np.isfinite(q_mean)

                if finite_mean_mask.any():
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

                    last_finite_idx = np.where(finite_mean_mask)[0][-1]
                    final_error = abs(q_mean[last_finite_idx] - reference_value)
                    ax.set_title(f"{title}\nDistance to Ref: {final_error:.3f}")
                else:
                    ax.set_title(f"{title}\nNo active {head_name} samples")
            else:
                ax.set_title(f"{title}\nNo active {head_name} samples")

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

        for row, col, indices, q_history, head_name, reference_value in plot_configs:
            ax = axes[row, col]  # type: ignore
            title = f"{head_name}: {group_label}={reference_value}"

            if not indices:  # Skip if no samples in this group
                ax.text(
                    0.5,
                    0.5,
                    "No samples",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                if col == 1 and middle_column_title:
                    ax.set_title(f"{middle_column_title}\n{title}")
                else:
                    ax.set_title(title)
                continue

            # Compute histograms for this reward group across epochs
            hist_data = []
            for q_epoch in q_history:
                q_epoch_arr = (
                    q_epoch.detach().cpu().numpy()
                    if hasattr(q_epoch, "detach")
                    else np.array(q_epoch)
                )
                q_subset = q_epoch_arr[indices].flatten()
                q_subset = q_subset[np.isfinite(q_subset)]
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
                if col == 1 and middle_column_title:
                    ax.set_title(f"{middle_column_title}\n{title}")
                else:
                    ax.set_title(title)
                ax.grid(True, alpha=0.3)
                plt.colorbar(im, ax=ax, label="Percentage (%)")


def plot_Qv_horizon(
    q_place: torch.Tensor | np.ndarray,
    q_select: torch.Tensor | np.ndarray,
    outcome: torch.Tensor | np.ndarray,
    steps_to_terminal: torch.Tensor | np.ndarray,
    fig_num: int = 5,
    DISPLAY_PLOT: bool = True,
    position: tuple[int, int] | None = (900, 0),
    experiment_name: str = "",
    FREQ_EPOCH_SAVING: int = -1,
    FOLDER_SAVE: str = "./",
    FIG_NAME=lambda epoch: f"{datetime.now().strftime('%Y%m%d_%H%M')}-qv_horizon_{epoch:04d}.svg",
    current_epoch: int = 0,
    SAVEFIG_DPI: int = 1000,
) -> None:
    """Plot current-epoch Q-values as a function of distance to terminal state.

    The plot aggregates the current batch by player-perspective outcome and
    horizon (steps to terminal), showing how each head values states at
    different depths from the end of the game.
    """

    def _to_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
        if hasattr(values, "detach"):
            return values.detach().cpu().numpy()
        return np.array(values)

    q_place_np = _to_numpy(q_place).reshape(-1)
    q_select_np = _to_numpy(q_select).reshape(-1)
    outcome_np = np.rint(_to_numpy(outcome).reshape(-1)).astype(int)
    steps_np = np.rint(_to_numpy(steps_to_terminal).reshape(-1)).astype(int)

    batch_size = min(
        q_place_np.shape[0],
        q_select_np.shape[0],
        outcome_np.shape[0],
        steps_np.shape[0],
    )
    if batch_size == 0:
        return

    q_place_np = q_place_np[:batch_size]
    q_select_np = q_select_np[:batch_size]
    outcome_np = outcome_np[:batch_size]
    steps_np = steps_np[:batch_size]

    experiment_name = f"{experiment_name}-{fig_num}"
    if plt.fignum_exists(experiment_name):
        fig = plt.figure(experiment_name)
        fig.clf()
    else:
        fig = None

    if fig is None:
        fig = plt.figure(experiment_name, figsize=(15, 9), constrained_layout=True)

    if position is not None:
        try:
            manager = fig.canvas.manager  # type: ignore
            manager.window.wm_geometry(f"+{position[0]}+{position[1]}")  # type: ignore
        except:
            pass

    axes = fig.subplots(2, 1, sharex=True)
    fig.suptitle(
        "Q-value Horizon Profile by Player Perspective\n"
        "Marker area scales with sample count; band = ±1 std",
        fontsize=14,
    )

    palette = {
        -1: {
            "color": "#E69F00",  # orange
            "marker": "o",
            "label": "Outcome = -1",
        },
        0: {
            "color": "#4D4D4D",  # graphite
            "marker": "s",
            "label": "Outcome = 0",
        },
        1: {
            "color": "#009E73",  # teal-green
            "marker": "^",
            "label": "Outcome = +1",
        },
    }

    head_configs = [
        (axes[0], q_place_np, "Q_place Horizon"),
        (axes[1], q_select_np, "Q_select Horizon"),
    ]

    finite_values = np.concatenate(
        [q_place_np[np.isfinite(q_place_np)], q_select_np[np.isfinite(q_select_np)]]
    )
    if finite_values.size > 0:
        q_min = finite_values.min()
        q_max = finite_values.max()
        y_pad = max(0.1, 0.05 * (q_max - q_min)) if q_max != q_min else 0.5
        y_limits = (q_min - y_pad, q_max + y_pad)
    else:
        y_limits = (-1.2, 1.2)

    for ax, q_values, title in head_configs:  # type: ignore
        for outcome_value, style in palette.items():
            outcome_mask = outcome_np == outcome_value
            if not outcome_mask.any():
                continue

            unique_steps = np.sort(np.unique(steps_np[outcome_mask]))
            plotted_steps = []
            q_mean = []
            q_std = []
            q_count = []

            for step in unique_steps:
                step_mask = outcome_mask & (steps_np == step)
                step_values = q_values[step_mask]
                step_values = step_values[np.isfinite(step_values)]
                if step_values.size == 0:
                    continue
                plotted_steps.append(step)
                q_mean.append(step_values.mean())
                q_std.append(step_values.std())
                q_count.append(step_values.size)

            if not q_mean:
                continue

            unique_steps = np.array(plotted_steps)
            q_mean_np = np.array(q_mean)
            q_std_np = np.array(q_std)
            q_count_np = np.array(q_count)
            marker_sizes = 40 + (20 * np.sqrt(q_count_np))

            ax.plot(
                unique_steps,
                q_mean_np,
                color=style["color"],
                linewidth=2.5,
                alpha=0.95,
                label=style["label"],
            )
            ax.scatter(
                unique_steps,
                q_mean_np,
                s=marker_sizes,
                color=style["color"],
                marker=style["marker"],
                edgecolors="white",
                linewidths=0.9,
                zorder=10,
            )
            ax.fill_between(
                unique_steps,
                q_mean_np - q_std_np,
                q_mean_np + q_std_np,
                color=style["color"],
                alpha=0.16,
            )

        ax.set_title(title)
        ax.set_ylabel("Mean Q-value")
        ax.set_ylim(y_limits)
        ax.grid(True, alpha=0.35, linestyle=":")
        ax.legend(loc="best", title="Perspective")

    axes[1].set_xlabel("Steps to terminal (0 = terminal state)")
    axes[1].set_xticks(np.sort(np.unique(steps_np)))

    if current_epoch % FREQ_EPOCH_SAVING == 0 and FREQ_EPOCH_SAVING != -1:
        plt.savefig(
            path.join(FOLDER_SAVE, FIG_NAME(current_epoch)),
            dpi=SAVEFIG_DPI,
            bbox_inches="tight",
        )

    if DISPLAY_PLOT:
        plt.draw()
        plt.pause(0.001)

    # Save the figure at regular intervals
    if current_epoch % FREQ_EPOCH_SAVING == 0 and FREQ_EPOCH_SAVING != -1:
        plt.savefig(
            path.join(FOLDER_SAVE, FIG_NAME(current_epoch)),
            dpi=SAVEFIG_DPI,
            bbox_inches="tight",
        )

    if DISPLAY_PLOT:
        plt.draw()
        plt.pause(0.001)
