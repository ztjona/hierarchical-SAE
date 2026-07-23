# -*- coding: utf-8 -*-

"""Generate parameter-sweep training scripts from trainRL.py.

Usage:
  run_trains.py <experiment_name>
  run_trains.py -h | --help

Arguments:
  <experiment_name>  Name for the experiment family (overrides the hardcoded
                     EXPERIMENT_NAME). PARAM_ITERATE and PARAMS stay as set
                     in the file.
"""
import re
from os import path
from datetime import datetime
from docopt import docopt


# INVALID results and before
# EXPERIMENT_NAME = "C01b_validate_N8"
# PARAM_ITERATE = "N_LAST_STATES_INIT"
# PARAMS = [4, 8, 16, 12]

# ----- VALID results
# Fixing bug of replay buffer size by epoch
# EXPERIMENT_NAME = "Aa_replay"
# PARAM_ITERATE = "NUM_EPOCHs_BUFFER"
# # The previous experiments can be considered to be set as 1.
# PARAMS = [2, 8, 128, 512, 1024, 64]

# VALID results
# Fixing bug of replay buffer size by epoch
# EXPERIMENT_NAME = "Ab_data"
# PARAM_ITERATE = "N_LAST_STATES_INIT"
# PARAMS = [4, 12, 16, 8]
# VALID results

# Fixing bug of replay buffer size by epoch
# EXPERIMENT_NAME = "Ac_fineShallow"
# PARAM_ITERATE = "LR"
# PARAMS = [1e-4, 3.5e-4, 5e-4, 1e-5, 5e-5, 5e-6, 7e-4]

# EXPERIMENT_NAME = "Ad_endgame"
# PARAM_ITERATE = "ENDGAME_FRACTION"
# PARAMS = [0.25, 0.5, 0.75]
# EXPERIMENT_NAME = "Ad_states_endgame"
# PARAM_ITERATE = "N_LAST_STATES_INIT"
# PARAMS = [3, 4, 5]

# EXPERIMENT_NAME = "JA_final"
# PARAM_ITERATE = "N_LAST_STATES_INIT"
# PARAMS = [2, 3, 4, 6, 12, 16]

# EXPERIMENT_NAME = "KA_coupled"
# PARAM_ITERATE = "N_LAST_STATES_INIT"
# PARAMS = [2, 3, 4, 6, 12, 16]

# EXPERIMENT_NAME = "Z_tempRegresive"

# Series O — unified-aux variant of the decoupled-autoreg trunk.
# Before running this sweep, edit trainRL.py:
#   TRANSITION_SCHEMA = "unified_autoreg"
#   ARCHITECTURE = QuartoCNNAutoregUnified  (or QuartoCNNAutoregUnifiedUnbound)
#   PLAYER_BOT_CLASS = Quarto_unified_bot
# OA_unifiedAux is the first sweep of code-version O. Compare to MB_final
# (matched LR/TAU/REWARD_FUNCTION) for the WR delta from input redesign alone.
# EXPERIMENT_NAME = "OA_unifiedAux"
# PARAM_ITERATE = "N_LAST_STATES_INIT"
# PARAMS = [2, 3, 4]

EXPERIMENT_NAME = "" # defined by CLI
PARAM_ITERATE = "N_LAST_STATES_INIT"
PARAMS = [2, 3, 4, 12]

# ----------------------------------------------------------------------
# Multi-param sweep mode (used when a single sweep needs to set more than
# one trainRL.py variable per variant — e.g. Ra_lossWeight sweeps both
# LOSS_ALPHA_PLACE and LOSS_ALPHA_SELECT). Leave at None to use the
# single-param mode above.
#
# Each item is a dict ``{param_name: value, ...}``. The variant suffix is
# built from the dict; you can override it via ``"_label": "ALPHA_1.0_3.0"``.
# When MULTI_PARAMS is set, PARAM_ITERATE / PARAMS are ignored.
#
# Example: Ra_lossWeight sweep — α grid over (place, select).
# MULTI_PARAMS = [
#     {"LOSS_ALPHA_PLACE": 1.0, "LOSS_ALPHA_SELECT": 1.0, "_label": "ALPHA_1.0_1.0"},
#     {"LOSS_ALPHA_PLACE": 1.0, "LOSS_ALPHA_SELECT": 3.0, "_label": "ALPHA_1.0_3.0"},
#     {"LOSS_ALPHA_PLACE": 0.3, "LOSS_ALPHA_SELECT": 1.0, "_label": "ALPHA_0.3_1.0"},
#     {"LOSS_ALPHA_PLACE": 0.1, "LOSS_ALPHA_SELECT": 1.0, "_label": "ALPHA_0.1_1.0"},
# ]
# Example: Sa_archScan sweep — ARCHITECTURE class (no quotes — the value is
# substituted verbatim, so the class must already be importable in trainRL.py).
# MULTI_PARAMS = [
#     {"ARCHITECTURE": "QuartoCNNAutoregUnifiedS1", "_label": "ARCH_S1_deepConv"},
#     {"ARCHITECTURE": "QuartoCNNAutoregUnifiedS2", "_label": "ARCH_S2_wideFC"},
#     {"ARCHITECTURE": "QuartoCNNAutoregUnifiedS4", "_label": "ARCH_S4_uniform512"},
# ]
MULTI_PARAMS = None

# Te_oracleAblation — Ve-series: ablate the minimax oracle mid-training and
# observe whether Q_select competence is sustained by MC supervision.
# Base recipe = Ta(1): unified_autoreg + QuartoCNNAutoregUnifiedS4 + depth-2
# minimax select target. EPOCHS bumped to 6000 so each variant has ≥2000
# post-ablation epochs.
# Wa_oracleStates — N_LAST_STATES sweep on the Ve recipe (Sa(3) S4 trunk +
# minimax depth=2 oracle always on, 10k epochs). Ve(4) already covers N=4;
# this sweep tests higher N to see if more oracle-supervised transitions per
# game helps or dilutes. Pre-T D1 showed OA(4) N=12 at chance without oracle;
# with oracle the picture may differ. [DONE 2026-06-04 — series-W.md]
#
# Xa_levers — 3-arm competence-lever screen [DONE 2026-06-08 — series-X.md].
# X(1) PLACE_WIN won; X(2) DEPTH_3 dominated; X(3) SEL_MARGIN rejected at λ=0.5
# (loss-scale artifact). Block removed; recoverable from git.
#
# Ya_hotHead — 4-arm λ_hot screen [DONE 2026-06-12 — series-Y.md]. The aux
# hot-piece BCE head broke the select wall: punishing avoidable 12%→1.64%
# (λ=1.0), Test-B hot-give 16%→4.5%, place intact, monotonic in λ. Block
# recoverable from git.
#
# Yb_hotChamp — 10k λ_hot promotion [DONE 2026-06-16 — series-Y.md]. Crowned
# Yb(3) λ_hot=1.0 seedB champion (avoidable 0.95%, place intact, λ=2.0 rejected).
# Block recoverable from git.
#
# Yc_nStates — N_LAST_STATES sweep on the CHAMPION recipe (S4Hot, λ_hot=1.0,
# λ_place_win=0.5, depth-2 minimax oracle always on, endgame anchor kept at
# ENDGAME_FRACTION=0.5 / N_LAST_STATES_ENDGAME=2). Same code-version as Y (S4Hot),
# only hyperparameters move → second-letter bump Yb→Yc. 6000-epoch screen; promote
# the winning N to 10k. Goal: attack the FORCED floor (now ~90% of residual loss)
# by training the place head on mid-game placement it currently never sees (N=4 is
# deep-endgame-only). N counts whole moves (estimate_steps_per_match = 2N-1), so
# N=16 ≈ the full game ("all positions"). Arms {6,8,12,16}; the N=4 control is
# Ya_hotHead(4)0610_HOT_1.0 @6k (forced 8.72%) — same recipe/protocol, no re-run.
#   1) N=6   — low-end resolution / cheap-win check just above the champion
#   2) N=8   — W's D1 optimum; minimal-cost midgame injection
#   3) N=12  — near-full; W's WR-tradeoff knee
#   4) N=16  — full game / max midgame coverage
# Prior (W, Ve recipe): N=8 is the D1 optimum & WR-tied, N≥12 strictly dominated
# on WR (−1.3–1.5pp) & D1 — but W NEVER measured forced rate. So gate PRIMARILY on
# the punishing-autopsy FORCED rate (+ raw forced count); guards: avoidable ≤~1%
# (give wall stays closed), Test-A + missed-win (place intact), WR vs BT/random, D1.
# A FLAT forced curve across N is itself decisive → the floor is a planning gap,
# pivot to the forcing-danger aux head. Seed noise on forced ≈0.8pp (Yb seedA↔B);
# if the cross-N spread is <~1pp, add a confirmation seed at the best N before any
# promotion. Single SEED=5 across arms (matches the Ya control; isolates the N axis).
_YC_BASE = {
    "ARCHITECTURE": "QuartoCNNAutoregUnifiedS4Hot",
    "USE_MINIMAX_SELECT_TARGET": True,
    "MINIMAX_SELECT_DEPTH": 2,
    "MINIMAX_DISABLE_AFTER_EPOCH": None,
    "LAMBDA_HOT": 1.0,
    "LAMBDA_PLACE_WIN": 0.5,
    "SEED": 5,
    "EPOCHS": "6_000",
}
MULTI_PARAMS = [
    {**_YC_BASE, "N_LAST_STATES_INIT": 6,  "_label": "N_6"},
    {**_YC_BASE, "N_LAST_STATES_INIT": 8,  "_label": "N_8"},
    {**_YC_BASE, "N_LAST_STATES_INIT": 12, "_label": "N_12"},
    {**_YC_BASE, "N_LAST_STATES_INIT": 16, "_label": "N_16"},
]

# Path to the original training script
TRAIN_SCRIPT = "trainRL.py"
OUTPUT_DIR = "train_scripts/"


def modify_param_in_file(file_path, param_name, param_value):
    """Modify a parameter value in the training script.

    Args:
        file_path: Path to the training script
        param_name: Name of the parameter to modify
        param_value: New value for the parameter

    Returns:
        Modified content as string
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find and replace the parameter line
    # Handles formats like: BATCH_SIZE = 512
    pattern = rf"^{param_name}\s*=\s*.*$"
    replacement = f"{param_name} = {param_value}"

    modified_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    return modified_content


def create_training_file(param_pairs, experiment_name):
    """Create training script with one or more modified parameters.

    Args:
        param_pairs: iterable of (name, value) tuples to substitute into
            trainRL.py. Values are written verbatim (no quoting), matching
            the existing behaviour for numeric and bare-identifier values.
        experiment_name: The name of the experiment, used for the filename.
    """
    print("=" * 80)
    print(f"Creating training file: {experiment_name}")
    for name, value in param_pairs:
        print(f"  {name} = {value}")
    print("=" * 80)

    with open(TRAIN_SCRIPT, "r", encoding="utf-8") as f:
        modified_content = f.read()
    for name, value in param_pairs:
        modified_content = modify_param_in_file_text(modified_content, name, value)

    exp_pattern = r'^EXPERIMENT_NAME\s*=\s*"([^"]+)"'
    exp_replacement = f'EXPERIMENT_NAME = "{experiment_name}"'
    modified_content = re.sub(
        exp_pattern, exp_replacement, modified_content, flags=re.MULTILINE
    )

    output_script_name = f"{experiment_name}.py"
    output_path = path.join(OUTPUT_DIR, output_script_name)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(modified_content)

    print(f"Created: {output_path}\n")


def modify_param_in_file_text(content, param_name, param_value):
    pattern = rf"^{param_name}\s*=\s*.*$"
    replacement = f"{param_name} = {param_value}"
    return re.sub(pattern, replacement, content, flags=re.MULTILINE)


def _build_variants():
    """Return list of (idx, label, [(name, value), ...]) for the configured sweep."""
    if MULTI_PARAMS:
        out = []
        for idx, spec in enumerate(MULTI_PARAMS, 1):
            spec = dict(spec)
            label = spec.pop("_label", None)
            pairs = list(spec.items())
            if label is None:
                label = "_".join(f"{k}_{v}" for k, v in pairs)
            out.append((idx, label, pairs))
        return out
    return [
        (idx, f"{PARAM_ITERATE}_{v}", [(PARAM_ITERATE, v)])
        for idx, v in enumerate(PARAMS, 1)
    ]


def main():
    """Main execution loop."""
    args = docopt(__doc__)
    global EXPERIMENT_NAME
    EXPERIMENT_NAME = args["<experiment_name>"]

    print(f"\n{'=' * 80}")
    print(f"CREATING TRAINING FILES")
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    if MULTI_PARAMS:
        print(f"Multi-param sweep: {len(MULTI_PARAMS)} variants")
    else:
        print(f"Parameter to iterate: {PARAM_ITERATE}")
        print(f"Values: {PARAMS}")
    print(f"{'=' * 80}\n")

    created_files = []
    run_commands = []
    nohup_lines = []  # parallel-safe nohup commands (thread-capped, logged)

    variants = _build_variants()
    for idx, label, pairs in variants:
        run_id = f"({idx:1d}){datetime.now():%m%d}"
        exp_variant_name = f"{EXPERIMENT_NAME}{run_id}_{label}"

        print(f"[{idx}/{len(variants)}] {label}")

        try:
            create_training_file(pairs, exp_variant_name)
            script_path = path.join(OUTPUT_DIR, f"{exp_variant_name}.py")
            created_files.append(script_path)

            if idx == len(variants):
                run_commands.append(f'./runpy.sh "{script_path}"')
            else:
                run_commands.append(f'./runpy.sh --no_echo "{script_path}" &')

            # nohup variant: thread-capped (1 BLAS thread / proc) so N parallel
            # PyTorch processes don't oversubscribe on a small-core machine.
            # Each process gets its own log file under logs/.
            log_path = f"logs/{exp_variant_name}.log"
            nohup_lines.append(
                "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                f'nohup ./runpy.sh --no_echo "{script_path}" '
                f'> "{log_path}" 2>&1 &'
            )
        except Exception as e:
            print(f"Failed to create file for {label}: {e}")

    print(f"\n{'=' * 80}")
    print("FILE CREATION COMPLETED")
    print(f"{'=' * 80}")
    print(f"\nCreated {len(created_files)} training files:")
    # for file in created_files:
    #     print(f"  - {file}")

    print(f"\n{'=' * 80}")
    print("TO RUN TRAINING (local, interactive):")
    print(f"{'=' * 80}")
    for command in run_commands:
        print(command)

    print(f"\n{'=' * 80}")
    print("TO RUN TRAINING (background, parallel, thread-capped, logged):")
    print(f"{'=' * 80}")
    print("mkdir -p logs")
    for line in nohup_lines:
        print(line)
    print("disown -a")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
