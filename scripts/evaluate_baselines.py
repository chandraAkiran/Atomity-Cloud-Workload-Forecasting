
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("reports/training_targets.csv")
HORIZON = 24
FOLD_ENDS = [62, 86, 110]


def persistence(history, horizon):
    return np.full(horizon, history[-1])


def moving_average(history, horizon, window=12):
    return np.full(horizon, np.mean(history[-window:]))


def linear_trend(history, horizon, window=24):
    recent = history[-window:]
    x = np.arange(len(recent))

    slope, intercept = np.polyfit(x, recent, 1)
    future_x = np.arange(
        len(recent), len(recent) + horizon
    )

    return slope * future_x + intercept


def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def main():
    targets = pd.read_csv(DATA_PATH)

    assert len(targets) == 110

    models = {
        "persistence": persistence,
        "moving_average_12": moving_average,
        "linear_trend_24": linear_trend,
    }

    results = []

    for target in [
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]:
        values = targets[target].to_numpy(dtype=float)

        for model_name, model_fn in models.items():
            for fold_number, end in enumerate(FOLD_ENDS, start=1):
                train_end = end - HORIZON

                history = values[:train_end]
                actual = values[train_end:end]

                prediction = model_fn(history, HORIZON)

                if target == "active_vm_count":
                    prediction = np.maximum(prediction, 0)
                else:
                    prediction = np.clip(prediction, 0, 100)

                results.append({
                    "target": target,
                    "model": model_name,
                    "fold": fold_number,
                    "mae": mae(actual, prediction),
                    "rmse": rmse(actual, prediction),
                })

    results_df = pd.DataFrame(results)

    summary = (
        results_df.groupby(["target", "model"])
        [["mae", "rmse"]]
        .mean()
        .reset_index()
    )
    
    print("\n=== BASELINE VALIDATION RESULTS ===")
    print(summary.to_string(index=False))

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    results_df.to_csv(
        output_dir / "baseline_fold_metrics.csv",
        index=False,
    )

    summary.to_csv(
        output_dir / "baseline_summary.csv",
        index=False,
    )

    print("\nSaved baseline metrics to reports/")

    
    print("\n=== FOLD-LEVEL RESULTS ===")
    print(
        results_df.to_string(index=False)
    )

    print("\n=== RELATIVE IMPROVEMENT OVER PERSISTENCE ===")

    cpu_summary = summary[
        summary["target"] == "aggregate_cpu_utilization"
    ].set_index("model")

    baseline_mae = cpu_summary.loc["persistence", "mae"]

    for model_name, row in cpu_summary.iterrows():
        improvement = (
            (baseline_mae - row["mae"]) / baseline_mae
        ) * 100

        print(f"{model_name}: {improvement:.2f}%")

    print("\n=== NORMALIZED CPU MAE ===")

    cpu_values = targets[
        "aggregate_cpu_utilization"
    ].to_numpy(dtype=float)

    for model_name, row in cpu_summary.iterrows():
        normalized_mae = (
            row["mae"] / np.mean(cpu_values)
        ) * 100

        print(f"{model_name}: {normalized_mae:.2f}%")

    print("\n=== HIGH-DEMAND VALIDATION ERROR ===")

    # Define peaks using training history only for each fold.
    peak_results = []

    for model_name, model_fn in models.items():
        for end in FOLD_ENDS:
            train_end = end - HORIZON
            history = cpu_values[:train_end]
            actual = cpu_values[train_end:end]

            prediction = np.clip(
                model_fn(history, HORIZON), 0, 100
            )

            # A threshold based only on the fold's history.
            peak_threshold = np.quantile(history, 0.90)
            peak_mask = actual >= peak_threshold

            if peak_mask.any():
                peak_mae = mae(
                    actual[peak_mask],
                    prediction[peak_mask],
                )
            else:
                peak_mae = np.nan

            peak_results.append({
                "model": model_name,
                "train_end": train_end,
                "peak_threshold": peak_threshold,
                "peak_points": int(peak_mask.sum()),
                "peak_mae": peak_mae,
            })

    peak_df = pd.DataFrame(peak_results)
    print(peak_df.to_string(index=False))

    peak_df.to_csv(
        output_dir / "baseline_peak_metrics.csv",
        index=False,
    )



if __name__ == "__main__":
    main()

