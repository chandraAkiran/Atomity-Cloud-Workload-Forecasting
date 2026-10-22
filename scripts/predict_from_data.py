
"""Generate a 24-step forecast directly from VM CPU readings."""

import argparse
from pathlib import Path

import pandas as pd

from scripts.generate_predictions import generate_predictions


def predict_from_data(data_dir, output_path):
    cpu_path = Path(data_dir) / "vm_cpu_readings.csv.gz"

    cpu = pd.read_csv(
        cpu_path,
        usecols=["timestamp", "vm_id", "avg_cpu"],
    )

    if cpu.empty:
        raise ValueError("CPU dataset is empty")

    targets = (
        cpu.groupby("timestamp")
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Reserve the final 24 timestamps as the holdout.
    horizon = 24

    if len(targets) < 12 + horizon:
        raise ValueError(
            "At least 36 timestamps are required: "
            "12 for training and 24 for the holdout."
        )

    training = targets.iloc[:-horizon].copy()

    predictions = generate_predictions(training)

    # Confirm that the generated timestamps match
    # the reserved holdout timestamp index.
    expected_timestamps = (
        targets["timestamp"].iloc[-horizon:].tolist()
    )

    if predictions["timestamp"].tolist() != expected_timestamps:
        raise ValueError(
            "Forecast timestamps do not match the holdout period"
        )

    predictions.to_csv(output_path, index=False)

    print(f"Training observations: {len(training)}")
    print(f"Forecast rows: {len(predictions)}")
    print(f"Saved predictions to: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="challenge_data",
        help="Directory containing vm_cpu_readings.csv.gz",
    )

    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Output prediction CSV path",
    )

    args = parser.parse_args()

    predict_from_data(args.data, args.output)


if __name__ == "__main__":
    main()

