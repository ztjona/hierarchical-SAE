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
# Xa_levers — 3-arm overnight competence-lever screen. Base = Ve(4) champion
# recipe (QuartoCNNAutoregUnifiedS4, N_LAST_STATES_INIT=4, depth-2 minimax
# oracle always on) at 6000 epochs (matches the Ve(1)@6k baseline). Each arm
# changes EXACTLY ONE thing; attribution is each arm's delta vs Ve(1)@6k.
# Motivated by the 2026-06-04 loss autopsy (analysis/competence_audit/REPORT.md):
# ~⅔ of vs-random losses are forced positions, ~⅓ avoidable select blunders,
# and the place head misses 6.7% of immediate wins. λ/margin are screen
# defaults (0.5/0.5) — tune on the winning lever's deep run. See series-X.md.
#   X(1) PLACE_WIN  : aux place-win hinge (LAMBDA_PLACE_WIN), depth 2.
#   X(2) DEPTH_3    : deeper select oracle (MINIMAX_SELECT_DEPTH=3), no aux.
#   X(3) SEL_MARGIN : aux select-margin hinge (LAMBDA_SEL_MARGIN), depth 2.
MULTI_PARAMS = [
    {
        "ARCHITECTURE": "QuartoCNNAutoregUnifiedS4",
        "USE_MINIMAX_SELECT_TARGET": True,
        "MINIMAX_SELECT_DEPTH": 2,
        "MINIMAX_DISABLE_AFTER_EPOCH": None,
        "N_LAST_STATES_INIT": 4,
        "EPOCHS": "6_000",
        "LAMBDA_PLACE_WIN": 0.5,
        "LAMBDA_SEL_MARGIN": 0.0,
        "_label": "PLACE_WIN",
    },
    {
        "ARCHITECTURE": "QuartoCNNAutoregUnifiedS4",
        "USE_MINIMAX_SELECT_TARGET": True,
        "MINIMAX_SELECT_DEPTH": 3,
        "MINIMAX_DISABLE_AFTER_EPOCH": None,
        "N_LAST_STATES_INIT": 4,
        "EPOCHS": "6_000",
        "LAMBDA_PLACE_WIN": 0.0,
        "LAMBDA_SEL_MARGIN": 0.0,
        "_label": "DEPTH_3",
    },
    {
        "ARCHITECTURE": "QuartoCNNAutoregUnifiedS4",
        "USE_MINIMAX_SELECT_TARGET": True,
        "MINIMAX_SELECT_DEPTH": 2,
        "MINIMAX_DISABLE_AFTER_EPOCH": None,
        "N_LAST_STATES_INIT": 4,
        "EPOCHS": "6_000",
        "LAMBDA_PLACE_WIN": 0.0,
        "LAMBDA_SEL_MARGIN": 0.5,
        "_label": "SEL_MARGIN",
    },
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
