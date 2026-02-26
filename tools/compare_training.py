# -*- coding: utf-8 -*-

"""
Training Comparison Script

This script compares training runs across different hyperparameter values,
plotting loss, win rate, Q-values, and other metrics for analysis.

Features:
- Compare multiple experiments simultaneously
- Include specific runs from previous experiments as baselines
- Interactive Plotly visualizations
- Automatic best model identification

Baselines are visualized with:
- Dashed lines (loss/win rate plots)
- Lower opacity and thinner lines (Q-value plots)
- Hollow markers (scatter plots)
- Experiment name prefix in legend

Author: z_tjona
Date: 2026
"""

import os
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import re
import colorsys
from tqdm import tqdm

# Configuration
# EXPERIMENT_NAME can be a single string or a list of experiment names to combine
# Examples:
#   EXPERIMENT_NAME = "B02replicate"  # Single experiment
#   EXPERIMENT_NAME = ["B02replicate", "B03_verLR"]  # Multiple experiments
# EXPERIMENT_NAME = "B02replicate"  # or ["B02replicate", "B03_verLR"] for combined
# EXPERIMENT_NAME = ["B02replicate", "B03_verLR"]  # for combined
EXPERIMENT_NAME = "C01b_validate_N8"

# BASELINEs: Include specific runs from previous experiments as reference points
# Format: List of dicts mapping experiment name to list of parameter values
# Examples:
#   [{"B02replicate": [1e-3, 5e-4]}, {"B03_verLR": [7e-4, 2e-3]}]
#   - Includes runs with LR=1e-3 and LR=5e-4 from B02replicate
#   - Includes runs with LR=7e-4 and LR=2e-3 from B03_verLR
#   [{"B02replicate": [1e-3]}]  # Include only one specific run
# Set to [] or None to disable baselines
#
# Baselines are shown with visual distinction:
#   - Dashed lines in line plots
#   - Hollow markers in scatter plots
#   - [Experiment Name] prefix in legend
BASELINEs = [{"B02replicate": [1e-3, 5e-4]}, {"B03_verLR": [7e-4, 2e-3]}]
# BASELINEs = []  # Disable baselines

CHECKPOINT_BASE = "./CHECKPOINTS/"
PARAM_NAME = "N_LAST_STATES_INIT"  # Parameter being varied


# Plot toggles - set to False to disable specific plots
PLOT_CONFIG = {
    "loss": True,
    "win_rate": True,
    "sample_efficiency": False,  # confusing
    "stability": False,  # confusing
    "q_values": True,
    "loss_vs_winrate": True,
    "heatmap": True,
}


def to_numpy(value):
    """Convert various types to numpy arrays, handling PyTorch tensors."""
    if hasattr(value, "detach"):  # PyTorch tensor
        return value.detach().cpu().numpy()
    elif hasattr(value, "__iter__") and not isinstance(value, (str, dict)):
        return np.array(value)
    else:
        return value


def generate_rainbow_colors(n):
    """Generate n evenly-spaced rainbow colors from blue to red."""
    colors = []
    for i in range(n):
        # HSV: Hue from 240° (blue) to 0° (red), going through the rainbow
        # 240° = 0.667 in [0,1], 0° = 0.0
        hue = 0.667 - (i / max(n - 1, 1)) * 0.667  # From blue to red
        saturation = 0.85  # High saturation for vivid colors
        value = 0.90  # High value for brightness

        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

        # Convert to hex
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        colors.append(hex_color)

    return colors


def format_param_value(value):
    """Format parameter value intelligently.

    - Values >= 1: show as integers or decimals (4, 8, 12, 16)
    - Values < 0.001: scientific notation (5e-04, 1e-03)
    - Values 0.001-1: decimal notation (0.001, 0.5)
    """
    if value >= 1:
        # Show as integer if it's a whole number, otherwise 1 decimal
        if value == int(value):
            return f"{int(value)}"
        else:
            return f"{value:.1f}"
    elif value < 0.001:
        # Use scientific notation for very small values
        return f"{value:.0e}"
    else:
        # Use decimal notation for values between 0.001 and 1
        return f"{value:.4g}"


def extract_param_value(folder_name):
    """Extract parameter value from folder name based on PARAM_NAME."""
    # Pattern: Experiment(X)MMDD_PARAM_NAME_VALUE
    # Examples: "LR_1e-5", "N_LAST_STATES_INIT_8", "BATCH_SIZE_32"
    pattern = rf"{PARAM_NAME}_([0-9.e-]+)"
    match = re.search(pattern, folder_name)
    if match:
        return float(match.group(1))

    # Fallback to LR_ pattern for old experiments
    match = re.search(r"LR_([0-9.e-]+)", folder_name)
    if match:
        return float(match.group(1))
    return None


def find_experiment_folders(base_path, experiment_names):
    """Find all checkpoint folders matching the experiment pattern(s).

    Args:
        base_path: Base path to checkpoints
        experiment_names: Single experiment name string or list of experiment names

    Returns:
        List of folder dictionaries sorted by parameter value
    """
    checkpoint_path = Path(base_path)

    # Ensure experiment_names is a list
    if isinstance(experiment_names, str):
        experiment_names = [experiment_names]

    folders = []

    # Search for each experiment pattern
    for exp_name in experiment_names:
        pattern = f"{exp_name}*"
        for folder in checkpoint_path.glob(pattern):
            if folder.is_dir():
                param_value = extract_param_value(folder.name)
                if param_value is not None:
                    folders.append(
                        {
                            "path": folder,
                            "name": folder.name,
                            "param_value": param_value,
                            "experiment": exp_name,
                            "is_baseline": False,
                        }
                    )

    # Sort by parameter value
    folders.sort(key=lambda x: x["param_value"])
    return folders


def load_baseline_experiments(base_path, baselines):
    """Load specific baseline experiments from previous runs.

    Args:
        base_path: Base path to checkpoints
        baselines: List of dicts like [{"B02replicate": [1e-3, 5e-4]}, {"B03_verLR": [7e-4]}]
                   Each dict maps experiment name to list of parameter values to include

    Returns:
        List of folder dictionaries for baseline experiments
    """
    if not baselines:
        return []

    checkpoint_path = Path(base_path)
    baseline_folders = []

    for baseline_dict in baselines:
        for exp_name, param_values in baseline_dict.items():
            # Find all folders for this experiment
            pattern = f"{exp_name}*"
            for folder in checkpoint_path.glob(pattern):
                if folder.is_dir():
                    param_value = extract_param_value(folder.name)
                    # Only include if parameter value matches one of the specified values
                    if param_value is not None and param_value in param_values:
                        baseline_folders.append(
                            {
                                "path": folder,
                                "name": folder.name,
                                "param_value": param_value,
                                "experiment": exp_name,
                                "is_baseline": True,
                            }
                        )

    # Sort by parameter value
    baseline_folders.sort(key=lambda x: x["param_value"])
    return baseline_folders


def load_experiment_data(folder_info):
    """Load the pickle file from an experiment folder."""
    pkl_pattern = f"{folder_info['name']}.pkl"
    pkl_path = folder_info["path"] / pkl_pattern

    # If exact match not found, try to find any pkl file
    if not pkl_path.exists():
        pkl_files = list(folder_info["path"].glob("*.pkl"))
        if pkl_files:
            pkl_path = pkl_files[0]
        else:
            return None

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None


def plot_losses(all_data, folders):
    """Create interactive Plotly plot for loss comparison."""
    fig = go.Figure()

    # Generate rainbow colors based on number of variants
    colors = generate_rainbow_colors(len(folders))

    smoothing_window = 50  # Moving average window

    for idx, (folder_info, data) in enumerate(zip(folders, all_data)):
        if data is None:
            continue

        loss_values = data["loss_values"]["loss_values"]
        param_value = folder_info["param_value"]
        is_baseline = folder_info.get("is_baseline", False)
        exp_name = folder_info.get("experiment", "")

        # Apply smoothing
        if len(loss_values) > smoothing_window:
            smoothed = np.convolve(
                loss_values, np.ones(smoothing_window) / smoothing_window, mode="valid"
            )
        else:
            smoothed = loss_values

        x_values = list(range(len(smoothed)))

        # Visual distinction for baselines: dashed line, reduced opacity
        line_style = "dash" if is_baseline else "solid"
        opacity = 0.6 if is_baseline else 0.8
        width = 2.0 if is_baseline else 2.5
        name_prefix = f"[{exp_name}] " if is_baseline else ""
        param_str = format_param_value(param_value)

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=smoothed,
                mode="lines",
                name=f"{name_prefix}{PARAM_NAME}={param_str}",
                line=dict(color=colors[idx], width=width, dash=line_style),
                opacity=opacity,
                hovertemplate=f"<b>{name_prefix}{PARAM_NAME}={param_str}</b><br>Iter: %{{x}}<br>Loss: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Training Loss Comparison - {EXPERIMENT_NAME}",
        xaxis_title="Training Iteration",
        yaxis_title="Loss (Smoothed)",
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def plot_win_rates(all_data, folders):
    """Create interactive Plotly plots for win rate comparison against each rival.

    Optimized for exactly 2 rivals.
    """
    # Get all rival names from the first valid dataset
    rival_names = None
    for data in all_data:
        if data is not None and "win_rate" in data:
            rival_names = list(data["win_rate"].keys())
            break

    if rival_names is None:
        print("No win rate data found!")
        return None

    # Create subplots for 2 rivals
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"vs {rival}" for rival in rival_names],
        horizontal_spacing=0.12,
    )

    # Generate rainbow colors based on number of variants
    colors = generate_rainbow_colors(len(folders))

    smoothing_window = 100  # Moving average window for win rates

    for idx, (folder_info, data) in enumerate(zip(folders, all_data)):
        if data is None:
            continue

        param_value = folder_info["param_value"]
        is_baseline = folder_info.get("is_baseline", False)
        exp_name = folder_info.get("experiment", "")
        win_rate_data = data.get("win_rate", {})

        # Visual distinction for baselines
        line_style = "dash" if is_baseline else "solid"
        opacity = 0.6 if is_baseline else 0.8
        width = 2.0 if is_baseline else 2.5
        name_prefix = f"[{exp_name}] " if is_baseline else ""
        param_str = format_param_value(param_value)

        for rival_idx, rival_name in enumerate(rival_names):
            if rival_name in win_rate_data:
                win_rates = win_rate_data[rival_name]

                # Apply smoothing to win rates
                if len(win_rates) > smoothing_window:
                    smoothed_wr = np.convolve(
                        win_rates,
                        np.ones(smoothing_window) / smoothing_window,
                        mode="valid",
                    )
                    epochs = list(range(len(smoothed_wr)))
                else:
                    smoothed_wr = win_rates
                    epochs = list(range(len(win_rates)))

                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=smoothed_wr,
                        mode="lines",
                        name=f"{name_prefix}{PARAM_NAME}={param_str}",
                        line=dict(color=colors[idx], width=width, dash=line_style),
                        opacity=opacity,
                        hovertemplate=f"<b>{name_prefix}{PARAM_NAME}={param_str}</b><br>Epoch: %{{x}}<br>Win Rate: %{{y:.2%}}<extra></extra>",
                        showlegend=(rival_idx == 0),  # Only show legend once
                    ),
                    row=1,
                    col=rival_idx + 1,
                )

    # Add 50% reference line to each subplot
    for rival_idx in range(2):
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            row=1,
            col=rival_idx + 1,
        )

    fig.update_layout(
        title_text=f"Win Rate Comparison - {EXPERIMENT_NAME}",
        template="plotly_white",
        width=1400,  # Optimized for 2 rivals (700px each)
        height=600,
        hovermode="x unified",
    )

    # Update y-axes
    for rival_idx in range(2):
        fig.update_yaxes(
            title_text="Win Rate",
            range=[0, 1],
            tickformat=".0%",
            row=1,
            col=rival_idx + 1,
        )
        fig.update_xaxes(title_text="Epoch", row=1, col=rival_idx + 1)

    return fig


def plot_sample_efficiency(all_data, folders):
    """Plot performance at key early milestones.

    Optimized for exactly 2 rivals.
    """
    milestones = [100, 500, 1000, 2000, 5000, 10000]

    # Get rival names
    rival_names = None
    for data in all_data:
        if data is not None and "win_rate" in data:
            rival_names = list(data["win_rate"].keys())
            break

    if not rival_names:
        return None

    colors = generate_rainbow_colors(len(folders))

    # Create subplots for 2 rivals
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"vs {rival}" for rival in rival_names],
        horizontal_spacing=0.12,
    )

    for rival_idx, rival_name in enumerate(rival_names):
        for idx, (folder_info, data) in enumerate(zip(folders, all_data)):
            if data is None:
                continue

            param_value = folder_info["param_value"]
            win_rates = data.get("win_rate", {}).get(rival_name, [])
            param_str = format_param_value(param_value)

            # Extract win rates at milestones
            milestone_wrs = []
            milestone_epochs = []
            for milestone in milestones:
                if milestone < len(win_rates):
                    milestone_wrs.append(win_rates[milestone])
                    milestone_epochs.append(milestone)

            if milestone_wrs:
                fig.add_trace(
                    go.Scatter(
                        x=milestone_epochs,
                        y=milestone_wrs,
                        mode="lines+markers",
                        name=f"{PARAM_NAME}={param_str}",
                        line=dict(color=colors[idx], width=2.5),
                        marker=dict(size=8),
                        opacity=0.8,
                        showlegend=(rival_idx == 0),
                    ),
                    row=1,
                    col=rival_idx + 1,
                )

        # Add reference line
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            row=1,
            col=rival_idx + 1,
        )

    fig.update_layout(
        title_text=f"Sample Efficiency - {EXPERIMENT_NAME}",
        template="plotly_white",
        width=1400,  # Optimized for 2 rivals
        height=600,
        hovermode="x unified",
    )

    for rival_idx in range(2):
        fig.update_yaxes(
            title_text="Win Rate",
            range=[0, 1],
            tickformat=".0%",
            row=1,
            col=rival_idx + 1,
        )
        fig.update_xaxes(title_text="Epoch", type="log", row=1, col=rival_idx + 1)

    return fig


def plot_stability(all_data, folders):
    """Plot training stability using rolling standard deviation.

    Optimized for exactly 2 rivals.
    """
    # Get rival names
    rival_names = None
    for data in all_data:
        if data is not None and "win_rate" in data:
            rival_names = list(data["win_rate"].keys())
            break

    if not rival_names:
        return None

    colors = generate_rainbow_colors(len(folders))

    # Create subplots for 2 rivals
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"vs {rival} (Stability)" for rival in rival_names],
        horizontal_spacing=0.12,
    )

    window = 100  # Rolling window for std calculation

    for rival_idx, rival_name in enumerate(rival_names):
        for idx, (folder_info, data) in enumerate(zip(folders, all_data)):
            if data is None:
                continue

            param_value = folder_info["param_value"]
            win_rates = data.get("win_rate", {}).get(rival_name, [])
            param_str = format_param_value(param_value)

            if len(win_rates) > window:
                # Calculate rolling standard deviation
                rolling_std = []
                for i in range(window, len(win_rates)):
                    std = np.std(win_rates[i - window : i])
                    rolling_std.append(std)

                epochs = list(range(window, len(win_rates)))

                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=rolling_std,
                        mode="lines",
                        name=f"{PARAM_NAME}={param_str}",
                        line=dict(color=colors[idx], width=2.5),
                        opacity=0.8,
                        hovertemplate=f"<b>{PARAM_NAME}={param_str}</b><br>Epoch: %{{x}}<br>Std Dev: %{{y:.3f}}<extra></extra>",
                        showlegend=(rival_idx == 0),
                    ),
                    row=1,
                    col=rival_idx + 1,
                )

    fig.update_layout(
        title_text=f"Training Stability (Rolling StdDev, window={window}) - {EXPERIMENT_NAME}",
        template="plotly_white",
        width=1400,  # Optimized for 2 rivals
        height=600,
        hovermode="x unified",
    )

    for rival_idx in range(2):
        fig.update_yaxes(title_text="Standard Deviation", row=1, col=rival_idx + 1)
        fig.update_xaxes(title_text="Epoch", row=1, col=rival_idx + 1)

    return fig


def plot_q_values(all_data, folders):
    """Plot Q-value evolution over training, grouped by target reward."""
    # Check if Q-value data exists
    has_q_data = any(
        data is not None and "q_values_history" in data for data in all_data
    )

    if not has_q_data:
        print("  ⚠ No Q-value data found in pickle files")
        return None

    colors = generate_rainbow_colors(len(folders))

    # Create subplots for place and select Q-values
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Q-values (Place)", "Q-values (Select)"],
        horizontal_spacing=0.15,
    )

    for idx, (folder_info, data) in enumerate(zip(folders, all_data)):
        if data is None or "q_values_history" not in data:
            continue

        param_value = folder_info["param_value"]
        is_baseline = folder_info.get("is_baseline", False)
        exp_name = folder_info.get("experiment", "")
        q_history = data["q_values_history"]

        # Visual distinction for baselines
        base_opacity = 0.5 if is_baseline else 0.7
        base_width = 1.5 if is_baseline else 2.0
        name_prefix = f"[{exp_name}] " if is_baseline else ""
        param_str = format_param_value(param_value)

        # Get rewards if available (for grouping)
        rewards = None
        if "rewards" in q_history and len(q_history["rewards"]) > 0:
            rewards = to_numpy(
                q_history["rewards"][0]
            )  # Rewards are same across epochs
        elif idx == 0:  # Only print warning once
            print(
                "  ⚠ No rewards found in Q-value history. Showing average across all targets."
            )
            print("    Re-run training to save rewards for grouped Q-value plots.")

        # Plot place Q-values grouped by target reward
        if "q_place" in q_history and q_history["q_place"]:
            q_place = q_history["q_place"]
            epochs = list(range(len(q_place)))

            if rewards is not None:
                # Group by target reward and plot separately
                for target_reward, line_style, target_name in [
                    (-1, "solid", "Loss"),
                    (0, "dot", "Draw"),
                    (1, "dash", "Win"),
                ]:
                    # Find indices for this reward
                    target_indices = [
                        i for i, r in enumerate(rewards) if round(r) == target_reward
                    ]

                    if target_indices:
                        # Average Q-values for this target group per epoch
                        # Important: filter indices to match each epoch's Q-value size
                        q_place_target = []
                        for q in q_place:
                            q_numpy = (
                                to_numpy(q) if hasattr(q, "__iter__") else np.array([q])
                            )
                            # Only use indices that are valid for this epoch's Q-values
                            valid_indices = [
                                i for i in target_indices if i < len(q_numpy)
                            ]
                            if valid_indices:
                                q_place_target.append(
                                    np.mean([q_numpy[i] for i in valid_indices])
                                )
                            else:
                                # No valid samples for this target in this epoch
                                q_place_target.append(np.nan)

                        fig.add_trace(
                            go.Scatter(
                                x=epochs,
                                y=q_place_target,
                                mode="lines",
                                name=f"{name_prefix}{PARAM_NAME}={param_str} ({target_name})",
                                line=dict(
                                    color=colors[idx], width=base_width, dash=line_style
                                ),
                                opacity=base_opacity,
                                showlegend=True,
                                legendgroup=f"lr_{idx}",
                            ),
                            row=1,
                            col=1,
                        )
            else:
                # Fallback: average all Q-values if rewards not available
                q_place_avg = [
                    np.mean(to_numpy(q)) if hasattr(q, "__iter__") else q
                    for q in q_place
                ]

                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=q_place_avg,
                        mode="lines",
                        name=f"{name_prefix}{PARAM_NAME}={param_str} (all)",
                        line=dict(color=colors[idx], width=base_width),
                        opacity=base_opacity,
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )

        # Plot select Q-values grouped by target reward
        if "q_select" in q_history and q_history["q_select"]:
            q_select = q_history["q_select"]
            epochs = list(range(len(q_select)))

            if rewards is not None:
                # Group by target reward and plot separately
                for target_reward, line_style, target_name in [
                    (-1, "solid", "Loss"),
                    (0, "dot", "Draw"),
                    (1, "dash", "Win"),
                ]:
                    # Find indices for this reward
                    target_indices = [
                        i for i, r in enumerate(rewards) if round(r) == target_reward
                    ]

                    if target_indices:
                        # Average Q-values for this target group per epoch
                        # Important: filter indices to match each epoch's Q-value size
                        q_select_target = []
                        for q in q_select:
                            q_numpy = (
                                to_numpy(q) if hasattr(q, "__iter__") else np.array([q])
                            )
                            # Only use indices that are valid for this epoch's Q-values
                            valid_indices = [
                                i for i in target_indices if i < len(q_numpy)
                            ]
                            if valid_indices:
                                q_select_target.append(
                                    np.mean([q_numpy[i] for i in valid_indices])
                                )
                            else:
                                # No valid samples for this target in this epoch
                                q_select_target.append(np.nan)

                        fig.add_trace(
                            go.Scatter(
                                x=epochs,
                                y=q_select_target,
                                mode="lines",
                                name=f"{name_prefix}{PARAM_NAME}={param_str} ({target_name})",
                                line=dict(
                                    color=colors[idx], width=base_width, dash=line_style
                                ),
                                opacity=base_opacity,
                                showlegend=False,
                                legendgroup=f"lr_{idx}",
                            ),
                            row=1,
                            col=2,
                        )
            else:
                # Fallback: average all Q-values if rewards not available
                q_select_avg = [
                    np.mean(to_numpy(q)) if hasattr(q, "__iter__") else q
                    for q in q_select
                ]

                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=q_select_avg,
                        mode="lines",
                        name=f"{name_prefix}{PARAM_NAME}={param_str} (all)",
                        line=dict(color=colors[idx], width=base_width),
                        opacity=base_opacity,
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

    # Add reference lines for target Q-values
    for col in [1, 2]:
        # Target lines at -1, 0, 1
        for target, color, name in [
            (-1, "red", "Target=-1"),
            (0, "gray", "Target=0"),
            (1, "green", "Target=1"),
        ]:
            fig.add_hline(
                y=target,
                line_dash="dash",
                line_color=color,
                opacity=0.5,
                line_width=1.5,
                annotation_text=name if col == 1 else None,
                annotation_position="right",
                row=1,
                col=col,
            )

    fig.update_layout(
        title_text=f"Q-Value Evolution - {EXPERIMENT_NAME}<br><sub>Grouped by target reward: solid=Loss(-1), dot=Draw(0), dash=Win(+1)</sub>",
        template="plotly_white",
        width=1200,
        height=600,
        hovermode="x unified",
    )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Average Q-Value", range=[-1.2, 1.2], row=1, col=1)
    fig.update_yaxes(title_text="Average Q-Value", range=[-1.2, 1.2], row=1, col=2)

    return fig


def plot_loss_vs_winrate(all_data, folders):
    """Plot loss vs win rate trade-off.

    Optimized for exactly 2 rivals.
    """
    # Get rival names
    rival_names = None
    for data in all_data:
        if data is not None and "win_rate" in data:
            rival_names = list(data["win_rate"].keys())
            break

    if not rival_names:
        return None

    colors = generate_rainbow_colors(len(folders))

    # Create subplots for 2 rivals
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"vs {rival}" for rival in rival_names],
        horizontal_spacing=0.15,
    )

    for rival_idx, rival_name in enumerate(rival_names):
        for idx, (folder_info, data) in enumerate(zip(folders, all_data)):
            if data is None:
                continue

            param_value = folder_info["param_value"]
            is_baseline = folder_info.get("is_baseline", False)
            exp_name = folder_info.get("experiment", "")
            name_prefix = f"[{exp_name}] " if is_baseline else ""
            param_str = format_param_value(param_value)

            # Get final loss (average of last 100 iterations)
            loss_values = data["loss_values"]["loss_values"]
            final_loss = np.mean(loss_values[-100:]) if len(loss_values) > 0 else 0

            # Get final win rate (average of last 100 epochs)
            win_rates = data.get("win_rate", {}).get(rival_name, [])
            final_wr = (
                np.mean(win_rates[-100:])
                if len(win_rates) >= 100
                else (np.mean(win_rates) if win_rates else 0)
            )

            # Calculate stability (std of last 100 epochs)
            stability = (
                1.0 / (1.0 + np.std(win_rates[-100:])) if len(win_rates) >= 100 else 1.0
            )

            # Visual distinction for baselines: hollow markers
            marker_symbol = "circle-open" if is_baseline else "circle"
            marker_line_width = 3 if is_baseline else 2

            fig.add_trace(
                go.Scatter(
                    x=[final_loss],
                    y=[final_wr],
                    mode="markers+text",
                    name=f"{name_prefix}{PARAM_NAME}={param_str}",
                    marker=dict(
                        color=colors[idx],
                        symbol=marker_symbol,
                        size=15 + stability * 20,  # Size represents stability
                        line=dict(
                            color="white" if not is_baseline else colors[idx],
                            width=marker_line_width,
                        ),
                    ),
                    text=f"{param_str}",
                    textposition="top center",
                    textfont=dict(size=10),
                    hovertemplate=f"<b>{name_prefix}{PARAM_NAME}={param_str}</b><br>Loss: %{{x:.4f}}<br>Win Rate: %{{y:.2%}}<br>Stability: {stability:.3f}<extra></extra>",
                    showlegend=(rival_idx == 0),
                ),
                row=1,
                col=rival_idx + 1,
            )

        # Add reference line at 50%
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            row=1,
            col=rival_idx + 1,
        )

    fig.update_layout(
        title_text=f"Loss vs Win Rate Trade-off - {EXPERIMENT_NAME}<br><sub>Marker size indicates stability</sub>",
        template="plotly_white",
        width=1400,  # Optimized for 2 rivals
        height=600,
    )

    for rival_idx in range(2):
        fig.update_xaxes(title_text="Final Loss", row=1, col=rival_idx + 1)
        fig.update_yaxes(
            title_text="Final Win Rate", tickformat=".0%", row=1, col=rival_idx + 1
        )

    return fig


def plot_performance_heatmap(all_data, folders):
    """Plot performance heatmap over time.

    Optimized for exactly 2 rivals.
    """
    # Get rival names
    rival_names = None
    for data in all_data:
        if data is not None and "win_rate" in data:
            rival_names = list(data["win_rate"].keys())
            break

    if not rival_names:
        return None

    # Create subplots for 2 rivals
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"vs {rival}" for rival in rival_names],
        horizontal_spacing=0.12,
    )

    for rival_idx, rival_name in enumerate(rival_names):
        # Prepare data matrix
        heatmap_data = []
        y_labels = []

        for folder_info, data in zip(folders, all_data):
            if data is None:
                continue

            param_value = folder_info["param_value"]
            win_rates = data.get("win_rate", {}).get(rival_name, [])

            if win_rates:
                # Apply smoothing for cleaner heatmap
                window = 50
                if len(win_rates) > window:
                    smoothed = np.convolve(
                        win_rates, np.ones(window) / window, mode="valid"
                    )
                else:
                    smoothed = win_rates

                heatmap_data.append(smoothed)
                y_labels.append(format_param_value(param_value))

        if heatmap_data:
            x_epochs = list(range(len(heatmap_data[0])))

            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data,
                    x=x_epochs,
                    y=y_labels,
                    colorscale="RdYlGn",  # Red-Yellow-Green
                    zmin=0,
                    zmax=1,
                    colorbar=(
                        dict(
                            title="Win Rate",
                            tickformat=".0%",
                        )
                        if rival_idx == 1  # Show on second (last) subplot
                        else None
                    ),
                    showscale=(rival_idx == 1),  # Show on second subplot
                    hovertemplate="Epoch: %{x}<br>LR: %{y}<br>Win Rate: %{z:.2%}<extra></extra>",
                ),
                row=1,
                col=rival_idx + 1,
            )

    fig.update_layout(
        title_text=f"Performance Heatmap Over Time - {EXPERIMENT_NAME}",
        template="plotly_white",
        width=1400,  # Optimized for 2 rivals (700px each)
        height=600,
    )

    for rival_idx in range(2):
        fig.update_xaxes(title_text="Epoch", row=1, col=rival_idx + 1)
        fig.update_yaxes(title_text=PARAM_NAME, row=1, col=rival_idx + 1)

    return fig


def find_best_models(all_data, folders):
    """Identify and print the best performing models."""
    print("\n" + "=" * 80)
    print("BEST MODELS ANALYSIS")
    print("=" * 80)

    # Analysis by final loss
    print("\n1. LOWEST FINAL LOSS:")
    print("-" * 80)
    final_losses = []
    for folder_info, data in zip(folders, all_data):
        if data is not None and "loss_values" in data:
            loss_values = data["loss_values"]["loss_values"]
            if len(loss_values) > 0:
                # Average last 100 iterations for stability
                final_loss = np.mean(loss_values[-100:])
                final_losses.append(
                    {
                        "name": folder_info["name"],
                        "param": folder_info["param_value"],
                        "final_loss": final_loss,
                    }
                )

    final_losses.sort(key=lambda x: x["final_loss"])
    for rank, item in enumerate(final_losses, 1):
        print(f"  {rank}. {item['name']}")
        print(
            f"     {PARAM_NAME}={format_param_value(item['param'])}, Final Loss={item['final_loss']:.6f}"
        )

    # Analysis by average win rate across all opponents
    print("\n2. BEST AVERAGE WIN RATE (ACROSS ALL OPPONENTS):")
    print("-" * 80)

    # Get rival names
    rival_names = None
    for data in all_data:
        if data is not None and "win_rate" in data:
            rival_names = list(data["win_rate"].keys())
            break

    if rival_names:
        avg_wr_results = []

        for folder_info, data in zip(folders, all_data):
            if data is not None and "win_rate" in data:
                # Calculate average win rate across all opponents
                all_final_wrs = []
                for rival in rival_names:
                    win_rates = data["win_rate"].get(rival, [])
                    if len(win_rates) > 0:
                        final_wr = np.mean(win_rates[-10:])  # Last 10 epochs
                        all_final_wrs.append(final_wr)

                if all_final_wrs:
                    avg_across_opponents = np.mean(all_final_wrs)
                    avg_wr_results.append(
                        {
                            "name": folder_info["name"],
                            "param": folder_info["param_value"],
                            "avg_wr": avg_across_opponents,
                            "individual_wrs": {
                                rival: (
                                    data["win_rate"].get(rival, [])[-10:]
                                    if len(data["win_rate"].get(rival, [])) > 0
                                    else []
                                )
                                for rival in rival_names
                            },
                        }
                    )

        avg_wr_results.sort(key=lambda x: x["avg_wr"], reverse=True)
        for rank, item in enumerate(avg_wr_results, 1):
            print(f"  {rank}. {item['name']}")
            individual_str = ", ".join(
                [
                    f"{rival}={np.mean(item['individual_wrs'][rival]):.2%}"
                    for rival in rival_names
                    if len(item["individual_wrs"][rival]) > 0
                ]
            )
            print(
                f"     {PARAM_NAME}={format_param_value(item['param'])}\n"
                f"     Average WR={item['avg_wr']:.2%} (avg of last 10 epochs)\n"
                f"     Individual: {individual_str}"
            )

    # Analysis by win rate against each rival
    print("\n3. BEST WIN RATES BY INDIVIDUAL RIVAL:")
    print("-" * 80)

    if rival_names:
        for rival in rival_names:
            print(f"\n  Against {rival}:")
            rival_results = []

            for folder_info, data in zip(folders, all_data):
                if data is not None and "win_rate" in data:
                    win_rates = data["win_rate"].get(rival, [])
                    if len(win_rates) > 0:
                        # Use average of last 10 epochs for stability
                        final_wr = np.mean(win_rates[-10:])
                        max_wr = np.max(win_rates)
                        rival_results.append(
                            {
                                "name": folder_info["name"],
                                "param": folder_info["param_value"],
                                "final_wr": final_wr,
                                "max_wr": max_wr,
                                "epoch_max": np.argmax(win_rates),
                            }
                        )

            rival_results.sort(key=lambda x: x["final_wr"], reverse=True)
            for rank, item in enumerate(rival_results[:3], 1):  # Top 3
                print(f"    {rank}. {item['name']}")
                print(
                    f"       {PARAM_NAME}={format_param_value(item['param'])}\n"
                    f"       Final WR={item['final_wr']:.2%} (avg of last 10 epochs)\n"
                    f"       Peak WR={item['max_wr']:.2%} (achieved at epoch {item['epoch_max']})"
                )

    # Overall best model (composite score)
    print("\n4. OVERALL BEST MODEL (Composite Score):")
    print("-" * 80)
    print("   Scoring: 40% Final Loss + 30% Avg Win Rate + 30% Max Win Rate")
    print("   Note: Avg Win Rate = mean of last 10 epochs across all rivals")

    composite_scores = []
    for folder_info, data in zip(folders, all_data):
        if data is None:
            continue

        # Get final loss (normalized, lower is better)
        loss_values = data["loss_values"]["loss_values"]
        final_loss = (
            np.mean(loss_values[-100:]) if len(loss_values) > 0 else float("inf")
        )

        # Get average win rates across all rivals
        win_rates_all = []
        max_win_rates = []
        if "win_rate" in data:
            for rival_name in data["win_rate"]:
                wr = data["win_rate"][rival_name]
                if len(wr) > 0:
                    win_rates_all.extend(wr[-10:])
                    max_win_rates.append(np.max(wr))

        avg_wr = np.mean(win_rates_all) if win_rates_all else 0
        max_wr = np.mean(max_win_rates) if max_win_rates else 0

        # Normalize loss (assuming range 0-1, invert for scoring)
        loss_score = max(0, 1 - final_loss)

        # Composite score
        composite = 0.4 * loss_score + 0.3 * avg_wr + 0.3 * max_wr

        composite_scores.append(
            {
                "name": folder_info["name"],
                "param": folder_info["param_value"],
                "score": composite,
                "final_loss": final_loss,
                "avg_wr": avg_wr,
                "max_wr": max_wr,
            }
        )

    composite_scores.sort(key=lambda x: x["score"], reverse=True)
    for rank, item in enumerate(composite_scores, 1):
        print(f"  {rank}. {item['name']}")
        print(
            f"     {PARAM_NAME}={format_param_value(item['param'])}, Overall Score={item['score']:.4f}"
        )
        print(
            f"     Final Loss={item['final_loss']:.6f}, "
            f"Avg WR={item['avg_wr']:.2%}, "
            f"Peak WR={item['max_wr']:.2%}"
        )

    print("\n" + "=" * 80)

    return composite_scores[0] if composite_scores else None


def main():
    """Main execution function."""
    # Normalize EXPERIMENT_NAME to list
    exp_names = (
        EXPERIMENT_NAME if isinstance(EXPERIMENT_NAME, list) else [EXPERIMENT_NAME]
    )
    exp_display = " + ".join(exp_names)
    exp_filename = "_".join(exp_names)

    print(f"\n{'='*80}")
    print(f"TRAINING COMPARISON: {exp_display}")
    print(f"{'='*80}\n")

    # Find all experiment folders
    folders = find_experiment_folders(CHECKPOINT_BASE, EXPERIMENT_NAME)

    if not folders:
        patterns = ", ".join([f"{name}*" for name in exp_names])
        print(f"No experiment folders found matching patterns: {patterns}")
        return

    print(f"Found {len(folders)} experiment variants:")
    for folder in folders:
        exp_tag = (
            f" [{folder.get('experiment', 'unknown')}]" if len(exp_names) > 1 else ""
        )
        print(
            f"  - {folder['name']}{exp_tag} ({PARAM_NAME}={format_param_value(folder['param_value'])})"
        )

    # Load baseline experiments if specified
    baseline_folders = load_baseline_experiments(CHECKPOINT_BASE, BASELINEs)
    if baseline_folders:
        print(f"\nIncluding {len(baseline_folders)} baseline experiments:")
        for folder in baseline_folders:
            print(
                f"  - [BASELINE] {folder['name']} [{folder.get('experiment', 'unknown')}] ({PARAM_NAME}={format_param_value(folder['param_value'])})"
            )
        # Merge baselines with current experiments
        folders = folders + baseline_folders

    # Load data from all experiments
    print(f"\nLoading data from {len(folders)} pickle files...")
    all_data = []
    loaded_count = 0

    with tqdm(total=len(folders), desc="Loading experiments", unit="file") as pbar:
        for folder in folders:
            data = load_experiment_data(folder)
            all_data.append(data)
            if data is not None:
                loaded_count += 1
                pbar.set_postfix_str(f"✓ {folder['name'][:30]}...")
            else:
                pbar.set_postfix_str(f"✗ {folder['name'][:30]}...")
            pbar.update(1)

    # Create results directory
    results_dir = Path("results") / exp_filename
    results_dir.mkdir(parents=True, exist_ok=True)

    # Count enabled plots
    enabled_plots = sum(1 for v in PLOT_CONFIG.values() if v)

    # Create plots
    print(f"\nGenerating {enabled_plots} comparison plots...")

    figures = []
    plot_progress = tqdm(total=enabled_plots, desc="Creating plots", unit="plot")

    # Loss plot
    if PLOT_CONFIG["loss"]:
        plot_progress.set_postfix_str("Generating loss plot...")
        fig_loss = plot_losses(all_data, folders)
        loss_png = results_dir / f"comparison_loss_{exp_filename}.png"
        fig_loss.write_image(str(loss_png), width=1200, height=600, scale=2)
        plot_progress.set_postfix_str(f"✓ Loss plot")
        plot_progress.update(1)
        figures.append(("Loss", fig_loss))

    # Win rate plots
    if PLOT_CONFIG["win_rate"]:
        plot_progress.set_postfix_str("Generating win rate plots...")
        fig_wr = plot_win_rates(all_data, folders)
        if fig_wr is not None:
            wr_png = results_dir / f"comparison_win_rate_{exp_filename}.png"
            # Fixed width for 2 rivals
            fig_wr.write_image(str(wr_png), width=1400, height=600, scale=2)
            plot_progress.set_postfix_str(f"✓ Win rate plots")
            figures.append(("Win Rate", fig_wr))
        plot_progress.update(1)

    # Sample efficiency plot
    if PLOT_CONFIG["sample_efficiency"]:
        plot_progress.set_postfix_str("Generating sample efficiency plot...")
        fig_eff = plot_sample_efficiency(all_data, folders)
        if fig_eff is not None:
            eff_png = results_dir / f"comparison_sample_efficiency_{exp_filename}.png"
            fig_eff.write_image(str(eff_png), width=1200, height=600, scale=2)
            plot_progress.set_postfix_str(f"✓ Sample efficiency plot")
            figures.append(("Sample Efficiency", fig_eff))
        plot_progress.update(1)

    # Stability plot
    if PLOT_CONFIG["stability"]:
        plot_progress.set_postfix_str("Generating stability plot...")
        fig_stab = plot_stability(all_data, folders)
        if fig_stab is not None:
            stab_png = results_dir / f"comparison_stability_{exp_filename}.png"
            fig_stab.write_image(str(stab_png), width=1200, height=600, scale=2)
            plot_progress.set_postfix_str(f"✓ Stability plot")
            figures.append(("Stability", fig_stab))
        plot_progress.update(1)

    # Q-values plot
    if PLOT_CONFIG["q_values"]:
        plot_progress.set_postfix_str("Generating Q-values plot...")
        fig_q = plot_q_values(all_data, folders)
        if fig_q is not None:
            q_png = results_dir / f"comparison_q_values_{exp_filename}.png"
            fig_q.write_image(str(q_png), width=1200, height=600, scale=2)
            plot_progress.set_postfix_str(f"✓ Q-values plot")
            figures.append(("Q-Values", fig_q))
        plot_progress.update(1)

    # Loss vs Win Rate trade-off plot
    if PLOT_CONFIG["loss_vs_winrate"]:
        plot_progress.set_postfix_str("Generating loss vs win rate plot...")
        fig_tradeoff = plot_loss_vs_winrate(all_data, folders)
        if fig_tradeoff is not None:
            tradeoff_png = (
                results_dir / f"comparison_loss_vs_winrate_{exp_filename}.png"
            )
            fig_tradeoff.write_image(str(tradeoff_png), width=1200, height=600, scale=2)
            plot_progress.set_postfix_str(f"✓ Loss vs win rate plot")
            figures.append(("Loss vs Win Rate", fig_tradeoff))
        plot_progress.update(1)

    # Performance heatmap
    if PLOT_CONFIG["heatmap"]:
        plot_progress.set_postfix_str("Generating performance heatmap...")
        fig_heat = plot_performance_heatmap(all_data, folders)
        if fig_heat is not None:
            heat_png = results_dir / f"comparison_heatmap_{exp_filename}.png"
            fig_heat.write_image(str(heat_png), width=1200, height=600, scale=2)
            plot_progress.set_postfix_str(f"✓ Performance heatmap")
            figures.append(("Heatmap", fig_heat))
        plot_progress.update(1)

    plot_progress.close()
    print(f"\n✓ All plots saved to: {results_dir}/")

    # Find best models
    best_model = find_best_models(all_data, folders)

    if best_model:
        print(f"\n{'='*80}")
        print(f"RECOMMENDATION: Use {best_model['name']}")
        print(f"    {PARAM_NAME}={format_param_value(best_model['param'])}")
        print(f"    Overall Score: {best_model['score']:.4f}")
        print(f"{'='*80}\n")

    # Show plots in browser
    if figures:
        print("Opening plots in browser...")
        for name, fig in figures:
            fig.show()


if __name__ == "__main__":
    main()
