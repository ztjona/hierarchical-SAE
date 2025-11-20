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
EXPERIMENT_NAME = "E04b"
PARAM_ITERATE = "N_LAST_STATES_FINAL"
PARAMS = [2, 4, 8, 12, 16]

# Path to the original training script
TRAIN_SCRIPT = "trainRL.py"


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
    import re

    pattern = rf"^{param_name}\s*=\s*.*$"
    replacement = f"{param_name} = {param_value}"

    modified_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    return modified_content


def create_training_file(param_value, experiment_suffix):
    """Create training script with modified parameter.

    Args:
        param_value: Value to set for the parameter
        experiment_suffix: Suffix to add to experiment name
    """
    print("=" * 80)
    print(f"Creating training file for {PARAM_ITERATE} = {param_value}")
    print("=" * 80)

    # Read and modify the training script
    modified_content = modify_param_in_file(TRAIN_SCRIPT, PARAM_ITERATE, param_value)
    print(f"Modified {PARAM_ITERATE} to {modified_content}.")
    # Also modify EXPERIMENT_NAME to include the parameter value
    import re

    exp_pattern = r'^EXPERIMENT_NAME\s*=\s*"([^"]+)"'
    exp_replacement = f'EXPERIMENT_NAME = "{EXPERIMENT_NAME}_{experiment_suffix}"'
    modified_content = re.sub(
        exp_pattern, exp_replacement, modified_content, flags=re.MULTILINE
    )

    # Create training script file
    output_script = f"trainRL_{experiment_suffix}.py"
    with open(output_script, "w", encoding="utf-8") as f:
        f.write(modified_content)

    print(f"Created: {output_script}")
    print(f"  - {PARAM_ITERATE} = {param_value}")
    print(f"  - EXPERIMENT_NAME = {EXPERIMENT_NAME}_{experiment_suffix}")
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

    for idx, param_value in enumerate(PARAMS, 1):
        experiment_suffix = f"{PARAM_ITERATE}{param_value}"

        print(
            f"[{idx}/{len(PARAMS)}] Creating file for {PARAM_ITERATE} = {param_value}"
        )

        try:
            create_training_file(param_value, experiment_suffix)
            created_files.append(f"trainRL_{experiment_suffix}.py")
        except Exception as e:
            print(f"Failed to create file for {param_value}: {e}")

    print(f"\n{'=' * 80}")
    print("FILE CREATION COMPLETED")
    print(f"{'=' * 80}")
    print(f"\nCreated {len(created_files)} training files:")
    for file in created_files:
        print(f"  - {file}")

    print(f"\n{'=' * 80}")
    print("TO RUN TRAINING WITH LOGGING:")
    print(f"{'=' * 80}")
    for idx, param_value in enumerate(PARAMS, 1):
        experiment_suffix = f"{PARAM_ITERATE}{param_value}"
        exp_name = f"{EXPERIMENT_NAME}_{experiment_suffix}"
        print(f"python trainRL_{experiment_suffix}.py 1> {exp_name}.log &")
        if idx == len(PARAMS):
            print(f"python trainRL_{experiment_suffix}.py | tee {exp_name}.log &")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
