
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("reports/training_targets.csv")
FOLD_ENDS = [62, 86, 110]
HORIZON = 24
LOOKBACK = 12
ALPHA = 10.0


def make_features(history, origin):
    """Features available at a single forecast origin."""
    recent = history[-LOOKBACK:]

    return np.array([
        recent[-1],
        recent.mean(),
        recent.std(),
        recent.min(),
        recent.max(),
        recent[-3:].mean(),
        recent[-6:].mean(),
        origin,
    ], dtype=float)


def make_supervised(history, horizon=HORIZON):
    """Build historical origin-to-future examples.

    Each example predicts a full 24-step future trajectory
    using only observations at or before its origin.
    """
    x_rows = []
    y_rows = []

    for end in range(LOOKBACK, len(history) - horizon + 1):
        past = history[:end]
        future = history[end:end + horizon]

        x_rows.append(make_features(past, end))
        y_rows.append(future)

    if not x_rows:
        raise ValueError("Insufficient training history")

    return np.vstack(x_rows), np.vstack(y_rows)


def ridge_forecast(history, horizon=HORIZON):
    x_train, y_train = make_supervised(history, horizon)

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=ALPHA),
    )

    model.fit(x_train, y_train)

    future_features = make_features(history, len(history))
    forecast = model.predict(future_features.reshape(1, -1))[0]

    return np.clip(forecast, 0, 100)


def moving_average_forecast(history, horizon=HORIZON):
    return np.full(horizon, np.mean(history[-LOOKBACK:]))


def evaluate():
    targets = pd.read_csv(DATA_PATH)
    assert len(targets) == 110

    values = targets["aggregate_cpu_utilization"].to_numpy(float)

    results = []

    for fold, end in enumerate(FOLD_ENDS, start=1):
        train_end = end - HORIZON

        history = values[:train_end]
        actual = values[train_end:end]

        predictions = {
            "moving_average_12": moving_average_forecast(history),
            "ridge_direct_24": ridge_forecast(history),
        }

        for name, predicted in predictions.items():
            results.append({
                "fold": fold,
                "model": name,
                "mae": np.mean(np.abs(actual - predicted)),
                "rmse": np.sqrt(np.mean((actual - predicted) ** 2)),
            })

    results_df = pd.DataFrame(results)

    print("\n=== RIDGE VS MOVING AVERAGE ===")
    print(results_df.to_string(index=False))

    print("\n=== MEAN VALIDATION PERFORMANCE ===")
    print(
        results_df.groupby("model")[["mae", "rmse"]]
        .mean()
        .to_string()
    )

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    results_df.to_csv(
        output_dir / "ridge_validation_metrics.csv",
        index=False,
    )


if __name__ == "__main__":
    evaluate()

