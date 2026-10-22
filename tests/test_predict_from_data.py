
import pandas as pd

from scripts.predict_from_data import predict_from_data


def test_smoke_dataset(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    records = []

    # 40 timestamps, only 3 VMs.
    for timestamp in range(0, 40 * 300, 300):
        for vm_id, cpu in [
            ("vm1", 10.0),
            ("vm2", 20.0),
            ("vm3", 30.0),
        ]:
            records.append({
                "timestamp": timestamp,
                "vm_id": vm_id,
                "avg_cpu": cpu,
            })

    pd.DataFrame(records).to_csv(
        data_dir / "vm_cpu_readings.csv.gz",
        index=False,
        compression="gzip",
    )

    output = tmp_path / "predictions.csv"

    predict_from_data(data_dir, output)

    predictions = pd.read_csv(output)

    assert predictions.shape == (24, 3)
    assert predictions["timestamp"].iloc[0] == 4800
    assert predictions["timestamp"].iloc[-1] == 11700
    assert (predictions["active_vm_count"] == 3).all()
    assert (
        predictions["aggregate_cpu_utilization"] == 20
    ).all()

