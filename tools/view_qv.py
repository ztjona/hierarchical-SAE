# -*- coding: utf-8 -*-
"""View Q-value progress plots from saved experiment results.

Usage:
    python tools/view_qv.py <experiment_name> <variation_num>

Examples:
    python tools/view_qv.py Ac_fineShallow 1
    python tools/view_qv.py Aa_replay 2
    python tools/view_qv.py Ab_data 3

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
from QuartoRL import plot_Qv_progress, plot_win_rate, plot_loss


class CPUUnpickler(pickle.Unpickler):
    """Unpickler that maps CUDA tensors to CPU."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(
                io.BytesIO(b), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


CHECKPOINTS_DIR = path.join(path.dirname(__file__), "..", "CHECKPOINTS")


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


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print()
        list_experiments()
        sys.exit(0)

    exp_name = sys.argv[1]
    var_num = int(sys.argv[2])

    # Optional: which plots to show (default: all)
    show_plots = sys.argv[3] if len(sys.argv) > 3 else "all"

    pkl_path = find_pkl(exp_name, var_num)
    if not pkl_path:
        print(f"No results found for {exp_name}({var_num})")
        list_experiments()
        sys.exit(1)

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

    rewards = qh["rewards"][0] if qh.get("rewards") else None
    if rewards is None:
        print("No reward data stored — cannot plot Q-value progress.")
        sys.exit(1)

    if show_plots in ("all", "qv"):
        plot_Qv_progress(
            qh,
            rewards,
            fig_num=4,
            DISPLAY_PLOT=True,
            PLOT_TYPE="hist",
            experiment_name=full_name,
            FREQ_EPOCH_SAVING=1,
            FOLDER_SAVE=results_dir,
            FIG_NAME=lambda epoch: f"{full_name}_qv.svg",
            current_epoch=1,
        )

    if show_plots in ("all", "wr"):
        plot_win_rate(
            *win_rate.items(),
            DISPLAY_PLOT=True,
            experiment_name=full_name,
            FREQ_EPOCH_SAVING=1,
            FOLDER_SAVE=results_dir,
            FIG_NAME=lambda epoch: f"{full_name}_win_rate.svg",
        )

    if show_plots in ("all", "loss"):
        plot_loss(
            loss_data,
            DISPLAY_PLOT=True,
            experiment_name=full_name,
            FREQ_EPOCH_SAVING=1,
            FOLDER_SAVE=results_dir,
            FIG_NAME=lambda epoch: f"{full_name}_loss.svg",
        )

    plt.show(block=True)


if __name__ == "__main__":
    main()
