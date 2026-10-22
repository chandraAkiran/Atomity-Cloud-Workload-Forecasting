
import numpy as np
import pandas as pd

from scripts.generate_predictions import generate_predictions


def make_training_data():
    timestamps = np.arange(110) * 300

    return pd.DataFrame({
        "timestamp": timestamps,
        "active_vm_count": np.full(110, 10000),
        "aggregate_cpu_utilization": np.linspace(8, 10, 110),
    })


def test_prediction_schema_and_horizon():
    training = make_training_data()
    predictions = generate_predictions(training)

    assert predictions.shape == (24, 3)
    assert predictions.columns.tolist() == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]

    assert predictions["timestamp"].iloc[0] == 33000
    assert predictions["timestamp"].iloc[-1] == 39900

    assert np.all(
        np.diff(predictions["timestamp"]) == 300
    )


def test_prediction_values():
    training = make_training_data()
    predictions = generate_predictions(training)

    assert predictions.notna().all().all()
    assert np.isfinite(predictions.to_numpy()).all()

    assert (predictions["active_vm_count"] == 10000).all()

    assert predictions[
        "aggregate_cpu_utilization"
    ].between(0, 100).all()

    expected_cpu = training[
        "aggregate_cpu_utilization"
    ].tail(12).mean()

    assert np.allclose(
        predictions["aggregate_cpu_utilization"],
        expected_cpu,
    )


def test_predictions_use_only_training_history():
    training = make_training_data()

    original_predictions = generate_predictions(training)

    # Change older training observations outside the
    # selected 12-step CPU forecasting window.
    modified_training = training.copy()
    modified_training.loc[
        :90, "aggregate_cpu_utilization"
    ] = 75.0

    modified_predictions = generate_predictions(
        modified_training
    )

    pd.testing.assert_frame_equal(
        original_predictions,
        modified_predictions,
    )


def test_invalid_training_length():
    training = make_training_data().iloc[:-1]

    try:
        generate_predictions(training)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for incorrect training length"
        )

