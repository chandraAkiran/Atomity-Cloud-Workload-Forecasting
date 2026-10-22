from pathlib import Path
import pandas as pd

DATA_DIR = Path("challenge_data")

cpu_path = DATA_DIR / "vm_cpu_readings.csv.gz"
metadata_path = DATA_DIR / "vm_metadata.csv.gz"

# Load the original Azure VM data
cpu = pd.read_csv(cpu_path)
metadata = pd.read_csv(metadata_path)

print("\n=== DATASET SHAPES ===")
print("CPU readings:", cpu.shape)
print("VM metadata:", metadata.shape)

print("\n=== CPU COLUMNS ===")
print(cpu.columns.tolist())

print("\n=== METADATA COLUMNS ===")
print(metadata.columns.tolist())

print("\n=== FIRST FIVE CPU RECORDS ===")
print(cpu.head())

print("\n=== FIRST FIVE METADATA RECORDS ===")
print(metadata.head())

print("\n=== MISSING VALUES ===")
print(cpu.isna().sum())
print(metadata.isna().sum())

print("\n=== DUPLICATE VM-TIMESTAMP RECORDS ===")
print(cpu.duplicated(subset=["vm_id", "timestamp"]).sum())

print("\n=== UNIQUE VMs ===")
print(cpu["vm_id"].nunique())

print("\n=== UNIQUE TIMESTAMPS ===")
print(cpu["timestamp"].nunique())

print("\n=== CPU STATISTICS ===")
print(cpu[["min_cpu", "max_cpu", "avg_cpu"]].describe())


# --------------------------------------------
# Timestamp cadence and VM activity
# --------------------------------------------

timestamps = sorted(cpu["timestamp"].unique())
intervals = pd.Series(timestamps).diff().dropna()

print("\n=== TIMESTAMP CADENCE ===")
print("First timestamp:", timestamps[0])
print("Last timestamp:", timestamps[-1])
print("Unique intervals:", intervals.unique())
print("All intervals equal 300 seconds:", intervals.eq(300).all())

print("\n=== VM ACTIVITY PER TIMESTAMP ===")

vm_counts = cpu.groupby("timestamp")["vm_id"].nunique()

print(vm_counts.describe())
print("All timestamps have 10,000 VMs:", vm_counts.eq(10000).all())

print("\n=== READINGS PER VM ===")

readings_per_vm = cpu.groupby("vm_id")["timestamp"].nunique()

print(readings_per_vm.describe())
print("All VMs have 134 readings:", readings_per_vm.eq(134).all())

# --------------------------------------------
# Official forecasting targets
# --------------------------------------------

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

print("\n=== FORECASTING TARGETS ===")
print(targets.head())
print(targets.describe())

# --------------------------------------------
# Chronological train/holdout split
# --------------------------------------------

assert len(targets) == 134, "Expected exactly 134 timestamps"

train = targets.iloc[:110].copy()
holdout = targets.iloc[110:].copy()

assert len(train) == 110
assert len(holdout) == 24

print("\n=== CHRONOLOGICAL SPLIT ===")
print("Training observations:", len(train))
print("Holdout observations:", len(holdout))
print("Last training timestamp:", train["timestamp"].iloc[-1])
print("First holdout timestamp:", holdout["timestamp"].iloc[0])

# Save training targets only.
# Do not export holdout target values for modeling.

output_dir = Path("reports")
output_dir.mkdir(exist_ok=True)

train.to_csv(output_dir / "training_targets.csv", index=False)

print("\nSaved: reports/training_targets.csv")
