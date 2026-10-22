
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("challenge_data/vm_cpu_readings.csv.gz")
PREDICTIONS_PATH = Path("predictions.csv")
OUTPUT_PATH = Path("reports/holdout_metrics.csv")

TRAIN_SIZE = 110
HORIZON = 24


def evaluate():
    # Load the already-generated, frozen predictions.
    predictions = pd.read_csv(PREDICTIONS_PATH)

    expected_columns = [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]

    assert predictions.columns.tolist() == expected_columns
    assert len(predictions) == HORIZON
    assert predictions["timestamp"].is_unique
    assert predictions["timestamp"].is_monotonic_increasing

    # The holdout data is read ONLY for final evaluation.
    cpu = pd.read_csv(
        DATA_PATH,
        usecols=["timestamp", "vm_id", "avg_cpu"],
    )

    actual_targets = (
        cpu.groupby("timestamp")
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .reset_index()
        .sort_values("timestamp")
    )

    assert len(actual_targets) == TRAIN_SIZE + HORIZON

    holdout = actual_targets.iloc[TRAIN_SIZE:].copy()

    assert np.array_equal(
        predictions["timestamp"].to_numpy(),
        holdout["timestamp"].to_numpy(),
    ), "Forecast timestamps must match the holdout exactly"

    results = []

    print("\n=== FINAL HOLDOUT EVALUATION ===")

    for target in [
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]:
        actual = holdout[target].to_numpy(float)
        predicted = predictions[target].to_numpy(float)

        assert np.isfinite(predicted).all()

        errors = actual - predicted

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        bias = np.mean(predicted - actual)

        normalized_mae = (
            100 * mae / np.mean(actual)
            if np.mean(actual) != 0
            else np.nan
        )

        print(f"\nTarget: {target}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"Normalized MAE: {normalized_mae:.3f}%")
        print(f"Mean forecast bias: {bias:.6f}")
        print(f"Actual minimum: {actual.min():.6f}")
        print(f"Actual maximum: {actual.max():.6f}")
        print(f"Forecast maximum: {predicted.max():.6f}")

        results.append({
            "target": target,
            "mae": mae,
            "rmse": rmse,
            "normalized_mae_percent": normalized_mae,
            "mean_forecast_bias": bias,
        })

        if target == "aggregate_cpu_utilization":
            # Diagnostic peak analysis, not model selection.
            peak_threshold = np.quantile(
                actual_targets.iloc[:TRAIN_SIZE][target],
                0.90,
            )

            peak_mask = actual >= peak_threshold

            print(f"Training-derived peak threshold: {peak_threshold:.6f}")
            print(f"Peak observations: {peak_mask.sum()}")

            if peak_mask.any():
                print(
                    "Peak MAE:",
                    f"{np.mean(np.abs(errors[peak_mask])):.6f}",
                )
                print(
                    "Maximum CPU underprediction:",
                    f"{np.max(errors):.6f} percentage points",
                )
            else:
                print("Peak MAE: not defined (no qualifying peaks)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved evaluation metrics to {OUTPUT_PATH}")
    print("Existing predictions.csv was not modified.")


if __name__ == "__main__":
    evaluate()

