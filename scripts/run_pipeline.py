
"""Run the reproducible forecasting workflow."""

import subprocess
import sys


def run_script(script):
    print(f"\n=== Running {script} ===", flush=True)
    subprocess.run(
        [sys.executable, script],
        check=True,
    )


def main():
    # Creates training targets from the original dataset.
    run_script("scripts/inspect_dataset.py")

    # Runs EDA using the training window only.
    run_script("scripts/explore_workload.py")

    # Validates the baseline and alternative model.
    run_script("scripts/evaluate_baselines.py")
    run_script("scripts/evaluate_ridge.py")

    # Generates the submission predictions using
    # only the first 110 target observations.
    run_script("scripts/generate_predictions.py")

    print("\nPipeline completed successfully.")
    print("Submission file: predictions.csv")
    print(
        "To evaluate the frozen holdout predictions, run:"
        "\npython scripts/evaluate_holdout.py"
    )


if __name__ == "__main__":
    main()

