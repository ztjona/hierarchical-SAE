# -*- coding: utf-8 -*-

"""
Python 3
30 / 10 / 2025
@author: z_tjona


"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

"Either mathematics is too big for the human mind or the human mind is more than a machine."
-Kurt Godël
"""
import re
from os import path
from datetime import datetime


EXPERIMENT_NAME = "B01_States"
PARAM_ITERATE = "N_LAST_STATES_FINAL"
PARAMS = [2, 4, 7, 10, 13, 16]

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


def create_training_file(param_value, experiment_name):
    """Create training script with modified parameter.

    Args:
        param_value: Value to set for the parameter
        experiment_name: The name of the experiment, used for the filename.
    """
    print("=" * 80)
    print(f"Creating training file for {PARAM_ITERATE} = {param_value}")
    print("=" * 80)

    # Read and modify the training script
    modified_content = modify_param_in_file(TRAIN_SCRIPT, PARAM_ITERATE, param_value)

    exp_pattern = r'^EXPERIMENT_NAME\s*=\s*"([^"]+)"'
    exp_replacement = f'EXPERIMENT_NAME = "{experiment_name}"'
    modified_content = re.sub(
        exp_pattern, exp_replacement, modified_content, flags=re.MULTILINE
    )

    # Create training script file
    output_script_name = f"train_{experiment_name}.py"
    output_path = path.join(OUTPUT_DIR, output_script_name)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(modified_content)

    print(f"Created: {output_path}")
    print(f"  - {PARAM_ITERATE} = {param_value}")
    print(f"  - EXPERIMENT_NAME = {experiment_name}")
    print("=" * 80 + "\n")


def main():
    """Main execution loop."""
    print(f"\n{'=' * 80}")
    print(f"CREATING TRAINING FILES")
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Parameter to iterate: {PARAM_ITERATE}")
    print(f"Values: {PARAMS}")
    print(f"{'=' * 80}\n")

    created_files = []
    run_commands = []

    for idx, param_value in enumerate(PARAMS, 1):
        run_id = f"{datetime.now():%m%d}-{idx:1d}"
        exp_name = f"{PARAM_ITERATE}_{run_id}_{param_value}"

        print(
            f"[{idx}/{len(PARAMS)}] Creating file for {PARAM_ITERATE} = {param_value}"
        )

        try:
            create_training_file(param_value, exp_name)
            script_path = path.join(OUTPUT_DIR, f"train_{exp_name}.py")
            created_files.append(script_path)

            if idx == len(PARAMS):
                run_commands.append(f"./runpy.sh {script_path}")
            else:
                run_commands.append(f"./runpy.sh {script_path} &")
        except Exception as e:
            print(f"Failed to create file for {param_value}: {e}")

    print(f"\n{'=' * 80}")
    print("FILE CREATION COMPLETED")
    print(f"{'=' * 80}")
    print(f"\nCreated {len(created_files)} training files:")
    # for file in created_files:
    #     print(f"  - {file}")

    print(f"\n{'=' * 80}")
    print("TO RUN TRAINING WITH LOGGING:")
    print(f"{'=' * 80}")
    for command in run_commands:
        print(command)

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
