
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_PATH = Path("reports/training_targets.csv")
OUTPUT_PATH = Path("predictions.csv")

TRAIN_SIZE = 110
HORIZON = 24
CADENCE_SECONDS = 300
CPU_WINDOW = 12


def generate_predictions(training_targets):
    """Forecast the next 24 timestamps using training data only."""

    required_columns = [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]

    if list(training_targets.columns) != required_columns:
        raise ValueError("Unexpected training target columns")

    if len(training_targets) != TRAIN_SIZE:
        raise ValueError(
            f"Expected {TRAIN_SIZE} training observations"
        )

    training_targets = training_targets.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    timestamps = training_targets["timestamp"].to_numpy()

    if not np.all(np.diff(timestamps) == CADENCE_SECONDS):
        raise ValueError("Training timestamps must have 300-second cadence")

    vm_history = training_targets[
        "active_vm_count"
    ].to_numpy(dtype=float)

    cpu_history = training_targets[
        "aggregate_cpu_utilization"
    ].to_numpy(dtype=float)

    if not np.isfinite(vm_history).all():
        raise ValueError("Non-finite VM counts")

    if not np.isfinite(cpu_history).all():
        raise ValueError("Non-finite CPU values")

    # Persistence baseline for active VM count.
    predicted_vm_count = np.full(
        HORIZON,
        vm_history[-1],
    )

    # Selected 12-step moving-average model for CPU.
    predicted_cpu = np.full(
        HORIZON,
        cpu_history[-CPU_WINDOW:].mean(),
    )

    future_timestamps = (
        timestamps[-1]
        + CADENCE_SECONDS * np.arange(1, HORIZON + 1)
    )

    predictions = pd.DataFrame({
        "timestamp": future_timestamps,
        "active_vm_count": predicted_vm_count,
        "aggregate_cpu_utilization": predicted_cpu,
    })

    predictions["active_vm_count"] = (
        predictions["active_vm_count"].round().astype(int)
    )

    predictions["aggregate_cpu_utilization"] = (
        predictions["aggregate_cpu_utilization"].clip(0, 100)
    )

    return predictions


def main():
    training_targets = pd.read_csv(TRAIN_PATH)

    predictions = generate_predictions(training_targets)

    assert len(predictions) == HORIZON
    assert predictions.notna().all().all()
    assert np.isfinite(
        predictions.select_dtypes(include="number").to_numpy()
    ).all()
    assert (predictions["active_vm_count"] >= 0).all()
    assert predictions[
        "aggregate_cpu_utilization"
    ].between(0, 100).all()

    predictions.to_csv(OUTPUT_PATH, index=False)

    print("\n=== FINAL FORECAST ===")
    print(predictions.to_string(index=False))
    print(f"\nSaved {len(predictions)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

