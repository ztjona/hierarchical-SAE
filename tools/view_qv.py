# -*- coding: utf-8 -*-
"""View Q-value progress plots from saved experiment results.

Usage:
    python tools/view_qv.py <experiment_name> [variation_num] [plot_type]

Examples:
    python tools/view_qv.py Ac_fineShallow 1
    python tools/view_qv.py Ac_fineShallow
    python tools/view_qv.py Aa_replay 2
    python tools/view_qv.py Ab_data 3

Notes:
    - Default plot type is qv (traditional Q-value histogram + horizon plot when metadata is available).
    - If variation_num is omitted, plots are generated for all matching experiment variations.
    - plot_type can be: qv, horizon, wr, loss, all.

If you don't know the exact folder name, run without arguments to list available experiments.
"""

import sys
import glob
import pickle
import io
from os import path

import torch
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, path.join(path.dirname(__file__), ".."))
from QuartoRL import plot_Qv_progress, plot_Qv_horizon, plot_win_rate, plot_loss


class CPUUnpickler(pickle.Unpickler):
    """Unpickler that maps CUDA tensors to CPU."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(
                io.BytesIO(b), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


CHECKPOINTS_DIR = path.join(path.dirname(__file__), "..", "CHECKPOINTS")

# Regenerated PNGs must fit under the 2576x2576 viewer limit.
# With figsize up to (16, 10), dpi=150 → 2400x1500 px, comfortably in range.
VIEW_DPI = 150


def list_experiments():
    """List all experiments with pickle files."""
    pkls = glob.glob(path.join(CHECKPOINTS_DIR, "**", "*.pkl"), recursive=True)
    if not pkls:
        print("No experiment results found in CHECKPOINTS/")
        return
    print("Available experiments:")
    for p in sorted(pkls):
        name = path.basename(p).replace(".pkl", "")
        size_mb = path.getsize(p) / (1024 * 1024)
        print(f"  {name}  ({size_mb:.1f} MB)")


def find_pkl(experiment_name: str, variation_num: int) -> str:
    """Find the pickle file for a given experiment and variation."""
    pattern = path.join(
        CHECKPOINTS_DIR, f"{experiment_name}({variation_num})*", "*.pkl"
    )
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    # Try without variation (e.g., base experiment)
    pattern = path.join(CHECKPOINTS_DIR, f"{experiment_name}", "*.pkl")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    # Try broader match
    pattern = path.join(CHECKPOINTS_DIR, f"{experiment_name}*", "*.pkl")
    matches = glob.glob(pattern)
    if matches:
        print(f"Found {len(matches)} matches:")
        for m in matches:
            print(f"  {path.basename(path.dirname(m))}")
        return matches[0]

    return ""


def find_pkls(experiment_name: str, variation_num: int | None = None) -> list[str]:
    """Find one or many pickle files for an experiment."""
    if variation_num is not None:
        pkl = find_pkl(experiment_name, variation_num)
        return [pkl] if pkl else []

    # No variation number: return all matching experiment variations.
    matches = glob.glob(path.join(CHECKPOINTS_DIR, f"{experiment_name}*", "*.pkl"))
    return sorted(matches)


def experiment_middle_title(full_name: str) -> str:
    """Build middle-column title with base experiment and hyperparameter."""
    import re

    base_match = re.match(r"^(.+?)\(\d+\)", full_name)
    if not base_match:
        return full_name

    base_exp = base_match.group(1)
    tail = full_name[base_match.end() :]
    tail = tail.lstrip("_")
    if "_" not in tail:
        return base_exp

    # Tail format is typically: MMDD_PARAM_NAME_PARAM_VALUE
    left, param_value = tail.rsplit("_", 1)
    if "_" not in left:
        return base_exp

    _, param_name = left.split("_", 1)
    return f"{base_exp}\n{param_name}={param_value}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print()
        list_experiments()
        sys.exit(0)

    exp_name = sys.argv[1]
    var_num = None

    # Optional: which plots to show (default: qv histogram + horizon if available)
    show_plots = "qv"
    if len(sys.argv) > 2:
        if sys.argv[2].isdigit():
            var_num = int(sys.argv[2])
            show_plots = sys.argv[3] if len(sys.argv) > 3 else "qv"
        else:
            show_plots = sys.argv[2]

    pkl_paths = find_pkls(exp_name, var_num)
    if not pkl_paths:
        if var_num is None:
            print(f"No results found for experiment prefix: {exp_name}")
        else:
            print(f"No results found for {exp_name}({var_num})")
        list_experiments()
        sys.exit(1)

    print(f"Found {len(pkl_paths)} matching experiment(s).")
    for pkl_path in pkl_paths:
        print(f"Loading: {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = CPUUnpickler(f).load()

        full_name = path.basename(pkl_path).replace(".pkl", "")
        folder = path.dirname(pkl_path)

        # Save plots in results/<experiment_name>/
        # Extract base experiment name (before the variation number)
        import re

        base_match = re.match(r"^(.+?)\(\d+\)", full_name)
        base_exp = base_match.group(1) if base_match else full_name
        results_dir = path.join(path.dirname(__file__), "..", "results", base_exp)
        if not path.exists(results_dir):
            import os

            os.makedirs(results_dir, exist_ok=True)

        qh = data["q_values_history"]
        loss_data = data["loss_values"]
        win_rate = data["win_rate"]

        print(f"Experiment: {full_name}")
        print(f"  Epochs with Q-data: {len(qh.get('q_place', []))}")
        print(f"  Loss points: {len(loss_data.get('loss_values', []))}")
        print(f"  Win rate rivals: {list(win_rate.keys())}")

        outcome = qh["outcome"][0] if qh.get("outcome") else None
        rewards = qh["rewards"][0] if qh.get("rewards") else None
        group_values = outcome if outcome is not None else rewards
        if group_values is None:
            print("No grouping data stored — cannot plot Q-value progress.")
            continue

        if show_plots in ("all", "qv"):
            plot_Qv_progress(
                qh,
                group_values,
                fig_num=4,
                DISPLAY_PLOT=True,
                PLOT_TYPE="hist",
                group_label="Outcome" if outcome is not None else "Reward",
                experiment_name=full_name,
                middle_column_title=experiment_middle_title(full_name),
                FREQ_EPOCH_SAVING=1,
                FOLDER_SAVE=results_dir,
                FIG_NAME=lambda epoch: f"{full_name}_qv.png",
                current_epoch=1,
                SAVEFIG_DPI=VIEW_DPI,
            )

        if show_plots in ("all", "qv", "horizon"):
            steps_to_terminal = (
                qh["steps_to_terminal"][0] if qh.get("steps_to_terminal") else None
            )
            if outcome is not None and steps_to_terminal is not None:
                plot_Qv_horizon(
                    qh["q_place"][-1],
                    qh["q_select"][-1],
                    outcome,
                    steps_to_terminal,
                    fig_num=5,
                    DISPLAY_PLOT=True,
                    experiment_name=full_name,
                    FREQ_EPOCH_SAVING=1,
                    FOLDER_SAVE=results_dir,
                    FIG_NAME=lambda epoch: f"{full_name}_qv_horizon.png",
                    current_epoch=1,
                    SAVEFIG_DPI=VIEW_DPI,
                )
            else:
                print(
                    "  Horizon metadata missing in this pickle "
                    "(needs outcome + steps_to_terminal). Skipping horizon plot."
                )

        if show_plots in ("all", "wr"):
            plot_win_rate(
                *win_rate.items(),
                DISPLAY_PLOT=True,
                experiment_name=full_name,
                FREQ_EPOCH_SAVING=1,
                FOLDER_SAVE=results_dir,
                FIG_NAME=lambda epoch: f"{full_name}_win_rate.png",
                SAVEFIG_DPI=VIEW_DPI,
            )

        if show_plots in ("all", "loss"):
            plot_loss(
                loss_data,
                DISPLAY_PLOT=True,
                experiment_name=full_name,
                FREQ_EPOCH_SAVING=1,
                FOLDER_SAVE=results_dir,
                FIG_NAME=lambda epoch: f"{full_name}_loss.png",
                SAVEFIG_DPI=VIEW_DPI,
            )

    plt.show(block=True)


if __name__ == "__main__":
    main()
