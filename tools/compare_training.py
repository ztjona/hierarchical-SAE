# -*- coding: utf-8 -*-

"""
Training Comparison Script - Pruned Edition

Compares training runs and plots loss + win rate by epoch.
Keep it simple: only two plots.

Author: z_tjona
Date: 2026
"""

import os
import pickle
import subprocess
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
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
# EXPERIMENT_NAME = "C01b_validate_N8" # previous results invalid!
# EXPERIMENT_NAME = "Aa_replay"
# PARAM_NAME = "NUM_EPOCHs_BUFFER"  # Parameter being varied
# EXPERIMENT_NAME = "Ab_data"
# PARAM_NAME = "N_LAST_STATES_INIT"
# EXPERIMENT_NAME = "Ac_fine"
# EXPERIMENT_NAME = "Ad_states_endgame"
# PARAM_NAME = "N_LAST_STATES_INIT"
# Before fixing replay buffer bug, results are invalid:
# EXPERIMENT_NAME = "GA_Bellman"
# PARAM_NAME = "N_LAST_STATES_INIT"
# EXPERIMENT_NAME = "HA_mask"
# PARAM_NAME = "N_LAST_STATES_INIT"

EXPERIMENT_NAME = "IA_unbound"
PARAM_NAME = "N_LAST_STATES_INIT"

# BASELINEs: Include specific runs from previous experiments as reference points
# Format: List of dicts mapping experiment name to list of parameter values
# Examples:
#   [{"B02replicate": [1e-3, 5e-4]}, {"B03_verLR": [7e-4, 2e-3]}]
#   - Includes runs with LR=1e-3 and LR=5e-4 from B02replicate
#   - Includes runs with LR=7e-4 and LR=2e-3 from B03_verLR
#   [{"B02replicate": [1e-3]}]  # Include only one specific run

BASELINEs = [
    {"B02replicate": [5e-4]},
    # {"B02replicate": [1e-3, 5e-4]},
    # {"B03_verLR": [7e-4, 2e-3]},
    {"B03_verLR": [2e-3]},
    {"Aa_replay": [8, 1024]},
    {"Ab_data": [8]},
    # {"Ab_data": [4, 8, 12]},
]
# BASELINEs = []  # Disable baselines

CHECKPOINT_BASE = "./CHECKPOINTS/"


# Plot toggles - set to False to disable specific plots
PLOT_CONFIG = {
    "loss": True,
    "win_rate": True,
}


def to_numpy(value):
    """Convert various types to numpy arrays, handling PyTorch tensors."""
    if hasattr(value, "detach"):  # PyTorch tensor
        return value.detach().cpu().numpy()
    elif hasattr(value, "__iter__") and not isinstance(value, (str, dict)):
        return np.array(value)
    else:
        return value


def _find_chromium():
    """Find a Chromium-based browser (Edge or Chrome) on Windows."""
    for prog_var in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
        base = os.environ.get(prog_var, "")
        if not base:
            continue
        for rel in (
            os.path.join("Google", "Chrome", "Application", "chrome.exe"),
            os.path.join("Microsoft", "Edge", "Application", "msedge.exe"),
        ):
            p = os.path.join(base, rel)
            if os.path.isfile(p):
                return p
    return None


def save_figure_png(fig, png_path, width=1200, height=600):
    """Save plotly figure as PNG via temp HTML + headless browser screenshot."""
    browser = _find_chromium()
    if not browser:
        html_path = Path(png_path).with_suffix(".html")
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"\u26a0 No browser found. Saved HTML instead: {html_path}")
        return

    tmp_html = Path(png_path).with_suffix(".tmp.html")
    try:
        fig.write_html(str(tmp_html), include_plotlyjs=True, auto_open=False)
        abs_png = str(Path(png_path).resolve())
        abs_html = str(tmp_html.resolve())
        subprocess.run(
            [
                browser,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                "--no-first-run",
                f"--screenshot={abs_png}",
                f"--window-size={width},{height}",
                "--force-device-scale-factor=2",
                "--virtual-time-budget=10000",
                abs_html,
            ],
            timeout=30,
            capture_output=True,
        )
        if not Path(png_path).exists():
            raise RuntimeError("Screenshot not created")
    except Exception as e:
        print(f"\u26a0 PNG export failed ({e}). Saving HTML instead.")
        html_path = Path(png_path).with_suffix(".html")
        fig.write_html(str(html_path), include_plotlyjs="cdn")
    finally:
        if tmp_html.exists():
            tmp_html.unlink()


def generate_rainbow_colors(n):
    """Generate n evenly-spaced rainbow colors from blue to red."""
    colors = []
    for i in range(n):
        hue = 0.667 - (i / max(n - 1, 1)) * 0.667  # blue to red
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.90)
        colors.append(
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        )
    return colors


def assign_colors(folders):
    """Assign colors: rainbow (blue→red) for experiments, Dark2 palette for baselines."""
    exp_indices = [i for i, f in enumerate(folders) if not f.get("is_baseline", False)]
    base_indices = [i for i, f in enumerate(folders) if f.get("is_baseline", False)]

    exp_colors = generate_rainbow_colors(len(exp_indices))
    base_palette = pc.qualitative.Dark2

    color_map = {}
    for j, idx in enumerate(exp_indices):
        color_map[idx] = exp_colors[j]
    for j, idx in enumerate(base_indices):
        color_map[idx] = base_palette[j % len(base_palette)]

    return [color_map[i] for i in range(len(folders))]


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


def extract_param_name_from_folder(folder_name):
    """Extract the actual parameter name from a folder name.

    Pattern: ExpName(idx)MMDD_PARAM_NAME_VALUE
    Examples:
      B02replicate(6)0121_LR_0.0005 -> LR
      Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8 -> NUM_EPOCHs_BUFFER
      Ab_data(4)0302_N_LAST_STATES_INIT_8 -> N_LAST_STATES_INIT
    """
    # Strip the trailing numeric value, then extract param name after the date
    match = re.match(r".*?\)\d{4}_(.+)_[0-9.e-]+$", folder_name, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def extract_trailing_numeric_value(folder_name):
    """Extract the trailing numeric token from a folder name.

    Examples:
      Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8 -> 8
      B02replicate(6)0121_LR_0.0005 -> 0.0005
    """
    match = re.search(
        r"_([0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)$", folder_name, re.IGNORECASE
    )
    if match:
        return float(match.group(1))
    return None


def numeric_value_matches(value, candidates, atol=1e-12, rtol=1e-9):
    """Return True if value matches any candidate using numeric tolerance."""
    for candidate in candidates:
        if np.isclose(value, float(candidate), atol=atol, rtol=rtol):
            return True
    return False


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
                    # Baselines should adapt to any parameter name.
                    # 1) Try current PARAM_NAME-based extractor (for backwards compatibility)
                    # 2) Fallback to trailing numeric token in folder name
                    param_value = extract_param_value(folder.name)
                    if param_value is None:
                        param_value = extract_trailing_numeric_value(folder.name)
                    # Only include if parameter value matches one of the specified values
                    if param_value is not None and numeric_value_matches(
                        param_value, param_values
                    ):
                        real_param_name = extract_param_name_from_folder(folder.name)
                        baseline_folders.append(
                            {
                                "path": folder,
                                "name": folder.name,
                                "param_value": param_value,
                                "param_name": real_param_name or PARAM_NAME,
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
    except Exception:
        pass

    # Fallback: file was pickle.dump'd with CUDA tensors.
    # torch.load's map_location only works for torch.save'd files.
    # We need a custom Unpickler to remap CUDA storages to CPU.
    try:
        import io
        import torch

        class _CpuUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch.storage" and name == "_load_from_bytes":
                    return lambda b: torch.load(
                        io.BytesIO(b), map_location="cpu", weights_only=False
                    )
                return super().find_class(module, name)

        with open(pkl_path, "rb") as f:
            data = _CpuUnpickler(f).load()
        return data
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None


def plot_losses(all_data, folders):
    """Create interactive Plotly plot for loss comparison."""
    fig = go.Figure()

    colors = assign_colors(folders)

    smoothing_window = 50  # Moving average window

    for idx, (folder_info, data) in enumerate(zip(folders, all_data)):
        if data is None:
            continue

        loss_values = to_numpy(data["loss_values"]["loss_values"])
        epoch_boundaries = to_numpy(data["loss_values"].get("epoch_values", []))
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

        # Convert iteration indices to epoch numbers using epoch_boundaries
        # epoch_boundaries[i] = last iteration index of epoch i
        if len(epoch_boundaries) > 0:
            iter_indices = np.arange(len(smoothed))
            # searchsorted: for each iteration, find which epoch it belongs to
            x_values = np.searchsorted(epoch_boundaries, iter_indices, side="left")
        else:
            x_values = list(range(len(smoothed)))

        # Visual distinction for baselines: dashed line, reduced opacity
        line_style = "dot" if is_baseline else "solid"
        opacity = 0.6 if is_baseline else 0.8
        width = 2.0 if is_baseline else 2.5
        name_prefix = f"[{exp_name}] " if is_baseline else ""
        label_param = folder_info.get("param_name", PARAM_NAME)
        param_str = format_param_value(param_value)

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=smoothed,
                mode="lines",
                name=f"{name_prefix}{label_param}={param_str}",
                line=dict(color=colors[idx], width=width, dash=line_style),
                opacity=opacity,
                hovertemplate=f"<b>{name_prefix}{label_param}={param_str}</b><br>Epoch: %{{x}}<br>Loss: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Training Loss Comparison - {EXPERIMENT_NAME}",
        xaxis_title="Epoch",
        yaxis_title="Loss (Smoothed)",
        hovermode="x unified",
        template="plotly_white",
        autosize=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=60, r=20, t=50, b=50),
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

    colors = assign_colors(folders)

    smoothing_window = 100  # Moving average window for win rates

    for idx, (folder_info, data) in enumerate(zip(folders, all_data)):
        if data is None:
            continue

        param_value = folder_info["param_value"]
        is_baseline = folder_info.get("is_baseline", False)
        exp_name = folder_info.get("experiment", "")
        win_rate_data = data.get("win_rate", {})

        # Visual distinction for baselines
        line_style = "dot" if is_baseline else "solid"
        opacity = 0.6 if is_baseline else 0.8
        width = 2.0 if is_baseline else 2.5
        name_prefix = f"[{exp_name}] " if is_baseline else ""
        label_param = folder_info.get("param_name", PARAM_NAME)
        param_str = format_param_value(param_value)

        for rival_idx, rival_name in enumerate(rival_names):
            if rival_name in win_rate_data:
                win_rates = to_numpy(win_rate_data[rival_name])

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
                        name=f"{name_prefix}{label_param}={param_str}",
                        line=dict(color=colors[idx], width=width, dash=line_style),
                        opacity=opacity,
                        hovertemplate=f"<b>{name_prefix}{label_param}={param_str}</b><br>Epoch: %{{x}}<br>Win Rate: %{{y:.2%}}<extra></extra>",
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
        autosize=True,
        hovermode="x unified",
        margin=dict(l=60, r=20, t=50, b=50),
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


def main():
    """Main execution function - generates loss and win rate plots."""
    exp_names = (
        EXPERIMENT_NAME if isinstance(EXPERIMENT_NAME, list) else [EXPERIMENT_NAME]
    )
    exp_display = " + ".join(exp_names)
    exp_filename = "_".join(exp_names)

    print(f"\n{'='*80}")
    print(f"TRAINING COMPARISON: {exp_display}")
    print(f"{'='*80}\n")

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

    baseline_folders = load_baseline_experiments(CHECKPOINT_BASE, BASELINEs)
    if baseline_folders:
        print(f"\nIncluding {len(baseline_folders)} baseline experiments:")
        for folder in baseline_folders:
            print(
                f"  - [BASELINE] {folder['name']} [{folder.get('experiment', 'unknown')}] ({PARAM_NAME}={format_param_value(folder['param_value'])})"
            )
        folders = folders + baseline_folders

    print(f"\nLoading data from {len(folders)} pickle files...")
    all_data = []

    with tqdm(total=len(folders), desc="Loading experiments", unit="file") as pbar:
        for folder in folders:
            data = load_experiment_data(folder)
            all_data.append(data)
            pbar.set_postfix_str(f"{'✓' if data else '✗'} {folder['name'][:30]}...")
            pbar.update(1)

    results_dir = Path("results") / exp_filename
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating loss and win rate plots...")

    fig_loss = plot_losses(all_data, folders)
    fig_wr = plot_win_rates(all_data, folders)

    # Save as PNG via headless browser screenshot
    loss_path = results_dir / f"comparison_loss_{exp_filename}.png"
    save_figure_png(fig_loss, loss_path)
    print(f"✓ Loss plot saved ({loss_path})")

    if fig_wr is not None:
        wr_path = results_dir / f"comparison_win_rate_{exp_filename}.png"
        save_figure_png(fig_wr, wr_path, width=1400)
        print(f"✓ Win rate plot saved ({wr_path})")

    print(f"\n✓ All plots saved to: {results_dir}/")
    print("\nOpening plots in browser...")
    fig_loss.show()
    if fig_wr is not None:
        fig_wr.show()


if __name__ == "__main__":
    main()
