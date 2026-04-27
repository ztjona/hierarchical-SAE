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


def _save_and_draw(
    fig,
    *,
    DISPLAY_PLOT: bool,
    FREQ_EPOCH_SAVING: int,
    FOLDER_SAVE: str,
    FIG_NAME,
    current_epoch: int,
    SAVEFIG_DPI: int,
) -> None:
    """Persist the intended figure and refresh the interactive window."""
    if FREQ_EPOCH_SAVING not in (-1, 0) and current_epoch % FREQ_EPOCH_SAVING == 0:
        fig.savefig(
            path.join(FOLDER_SAVE, FIG_NAME(current_epoch)),
            dpi=SAVEFIG_DPI,
            bbox_inches="tight",
        )

    if DISPLAY_PLOT:
        plt.draw()
        plt.pause(0.001)


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

    _save_and_draw(
        fig,
        DISPLAY_PLOT=DISPLAY_PLOT,
        FREQ_EPOCH_SAVING=FREQ_EPOCH_SAVING,
        FOLDER_SAVE=FOLDER_SAVE,
        FIG_NAME=FIG_NAME,
        current_epoch=current_epoch,
        SAVEFIG_DPI=SAVEFIG_DPI,
    )


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

    _save_and_draw(
        fig,
        DISPLAY_PLOT=DISPLAY_PLOT,
        FREQ_EPOCH_SAVING=FREQ_EPOCH_SAVING,
        FOLDER_SAVE=FOLDER_SAVE,
        FIG_NAME=FIG_NAME,
        current_epoch=current_epoch,
        SAVEFIG_DPI=SAVEFIG_DPI,
    )


def plot_Qv_horizon(
    q_place: torch.Tensor | np.ndarray,
    q_select: torch.Tensor | np.ndarray,
    outcome: torch.Tensor | np.ndarray,
    steps_to_terminal: torch.Tensor | np.ndarray,
    phase: torch.Tensor | np.ndarray | None = None,
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
    """Plot taken-action Q-value distributions by exact distance to terminal.

    Each panel contains the distribution of the stored taken-action Q-values for
    one head and one player-perspective outcome.

    **Joint schema** (``phase=None``): columns on the x-axis are exact
    ``steps_to_terminal`` buckets.

    **Decoupled-autoregressive schema** (``phase`` provided): the x-axis
    switches to *joint-turn* index so that Q_place and Q_select for the same
    player's turn are **vertically aligned**.  In the decoupled schema, even
    steps are always PHASE_PLACE and odd steps are always PHASE_SELECT.
    Joint-turn T = step // 2 for place, T = (step + 1) // 2 for select, so the
    paired actions (place step=2k, select step=2k−1) share the same column T=k.
    Tick labels show the place-step number for each joint turn.  The terminal
    turn (step=0) has no paired select, so that Q_select column is shown as
    white.
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

    phase_np: np.ndarray | None = None
    if phase is not None:
        phase_np = _to_numpy(phase).reshape(-1)[:batch_size].astype(int)

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

    axes = np.asarray(fig.subplots(2, 3, sharex=False, sharey=True))
    fig.suptitle(
        "Taken-Action Q-value Distributions by Horizon\n"
        "Each column is an exact steps-to-terminal bucket; colors sum to 100% within each step",
        fontsize=14,
    )

    hist_range = (-1.0, 1.0)
    outcome_columns = [(-1, "Outcome = -1"), (0, "Outcome = 0"), (1, "Outcome = +1")]
    hist_bins = 50
    head_rows = [
        ("Q_place", q_place_np),
        ("Q_select", q_select_np),
    ]
    heatmap_artist = None

    # Decoupled schema: precompute per-outcome-column turn mapping.
    # PHASE_PLACE=0 (even steps), PHASE_SELECT=1 (odd steps).
    # Joint-turn T: place step s → T = s // 2; select step s → T = (s + 1) // 2.
    _PHASE_PLACE = 0
    _PHASE_SELECT = 1
    col_info: list[dict] = []
    if phase_np is not None:
        for outcome_value, _ in outcome_columns:
            pm = (outcome_np == outcome_value) & (phase_np == _PHASE_PLACE)
            sm = (outcome_np == outcome_value) & (phase_np == _PHASE_SELECT)
            ps = np.sort(np.unique(steps_np[pm]).astype(int))  # even steps
            ss = np.sort(np.unique(steps_np[sm]).astype(int))  # odd steps
            pt = ps // 2
            st = (ss + 1) // 2
            parts = ([pt] if pt.size > 0 else []) + ([st] if st.size > 0 else [])
            all_T = (
                np.sort(np.unique(np.concatenate(parts)))
                if parts
                else np.array([], dtype=int)
            )
            T_to_L = {int(T): int(L) for L, T in enumerate(all_T)}
            col_info.append(
                {
                    "place_mask": pm,
                    "select_mask": sm,
                    "place_steps": ps,
                    "select_steps": ss,
                    "T_to_L": T_to_L,
                    "n_turns": len(all_T),
                    # tick labels are computed per-head at render time
                }
            )

    # Colormap with white for NaN columns (no data at that turn for this head).
    _cmap_nan = plt.cm.summer.copy()
    _cmap_nan.set_bad(color="white")

    for row, (head_name, q_values) in enumerate(head_rows):
        for col, (outcome_value, title) in enumerate(outcome_columns):
            ax = axes[row, col]

            if phase_np is not None:
                # ---- Decoupled schema: phase-filtered, turn-aligned ----
                info = col_info[col]
                n_turns = info["n_turns"]
                T_to_L = info["T_to_L"]

                if head_name == "Q_place":
                    active_mask = info["place_mask"]
                    active_steps = info["place_steps"]
                else:
                    active_mask = info["select_mask"]
                    active_steps = info["select_steps"]

                has_valid_samples = False
                if n_turns == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No valid samples",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                else:
                    # NaN init: columns without data stay NaN → rendered white.
                    hist_matrix = np.full((hist_bins, n_turns), np.nan, dtype=float)
                    for step in active_steps:
                        step_values = q_values[active_mask & (steps_np == step)]
                        step_values = step_values[np.isfinite(step_values)]
                        if step_values.size == 0:
                            continue
                        has_valid_samples = True
                        T = (
                            int(step) // 2
                            if head_name == "Q_place"
                            else (int(step) + 1) // 2
                        )
                        L = T_to_L[T]
                        hist, _ = np.histogram(
                            step_values, bins=hist_bins, range=hist_range
                        )
                        hist_matrix[:, L] = (hist / hist.sum()) * 100

                    if has_valid_samples:
                        heatmap_artist = ax.imshow(
                            hist_matrix,
                            aspect="auto",
                            origin="lower",
                            cmap=_cmap_nan,
                            interpolation="nearest",
                            extent=[-0.5, n_turns - 0.5, hist_range[0], hist_range[1]],
                            vmin=0,
                            vmax=100,
                        )
                        all_T = sorted(info["T_to_L"].keys())
                        if head_name == "Q_place":
                            _tick_labels = [str(int(T) * 2) for T in all_T]
                        else:
                            _tick_labels = [str(int(T) * 2 - 1) for T in all_T]
                        ax.set_xticks(range(n_turns))
                        ax.set_xticklabels(_tick_labels)
                        ax.set_xlim(-0.5, n_turns - 0.5)
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "No valid samples",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )

            else:
                # ---- Joint schema: original step-based logic ----
                mask = outcome_np == outcome_value
                if head_name == "Q_select":
                    # step=0 is the terminal placement — no piece selection follows.
                    mask &= steps_np > 0
                _active = steps_np[mask]
                subplot_steps = (
                    np.sort(np.unique(_active.astype(int)))
                    if _active.size > 0
                    else np.array([], dtype=int)
                )
                n_steps = max(subplot_steps.size, 1)
                hist_matrix = np.zeros((hist_bins, n_steps), dtype=float)
                has_valid_samples = False

                for step_idx, step in enumerate(subplot_steps):
                    step_values = q_values[mask & (steps_np == step)]
                    step_values = step_values[np.isfinite(step_values)]
                    if step_values.size == 0:
                        continue
                    has_valid_samples = True
                    hist, _ = np.histogram(
                        step_values, bins=hist_bins, range=hist_range
                    )
                    hist_matrix[:, step_idx] = (hist / hist.sum()) * 100

                if has_valid_samples:
                    heatmap_artist = ax.imshow(
                        hist_matrix,
                        aspect="auto",
                        origin="lower",
                        cmap="summer",
                        interpolation="nearest",
                        extent=[
                            subplot_steps[0] - 0.5,
                            subplot_steps[-1] + 0.5,
                            hist_range[0],
                            hist_range[1],
                        ],
                        vmin=0,
                        vmax=100,
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No valid samples",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

                if subplot_steps.size > 0:
                    ax.set_xticks(subplot_steps)
                    ax.set_xlim(subplot_steps[0] - 0.5, subplot_steps[-1] + 0.5)

            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{head_name}\nTaken-action Q-value")
            if row == 1:
                if phase_np is not None:
                    ax.set_xlabel("Place step  (0 = last;  Q_select uses place − 1)")
                else:
                    ax.set_xlabel("Steps to terminal  (0 = last move)")
            ax.set_ylim(-1, 1)
            ax.invert_xaxis()
            ax.grid(True, alpha=0.2, linestyle=":")

    if heatmap_artist is not None:
        fig.colorbar(
            heatmap_artist,
            ax=axes.ravel().tolist(),
            label="Within-step share (%)",
            shrink=0.96,
        )

    _save_and_draw(
        fig,
        DISPLAY_PLOT=DISPLAY_PLOT,
        FREQ_EPOCH_SAVING=FREQ_EPOCH_SAVING,
        FOLDER_SAVE=FOLDER_SAVE,
        FIG_NAME=FIG_NAME,
        current_epoch=current_epoch,
        SAVEFIG_DPI=SAVEFIG_DPI,
    )
