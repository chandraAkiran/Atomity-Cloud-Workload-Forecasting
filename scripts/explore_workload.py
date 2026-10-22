
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path("challenge_data")
OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cpu = pd.read_csv(DATA_DIR / "vm_cpu_readings.csv.gz")
metadata = pd.read_csv(DATA_DIR / "vm_metadata.csv.gz")

# Protect the holdout: restrict EDA to the first 110 timestamps.
timestamps = sorted(cpu["timestamp"].unique())
training_timestamps = timestamps[:110]

train_cpu = cpu[cpu["timestamp"].isin(training_timestamps)].copy()

print("Training CPU rows:", len(train_cpu))
print("Training timestamps:", train_cpu["timestamp"].nunique())

# 1. Fleet-wide CPU utilization over time
fleet_cpu = (
    train_cpu.groupby("timestamp")["avg_cpu"]
    .mean()
    .sort_index()
)

plt.figure(figsize=(11, 4))
plt.plot(fleet_cpu.index / 3600, fleet_cpu.values)
plt.xlabel("Elapsed time (hours)")
plt.ylabel("Mean CPU utilization (%)")
plt.title("Fleet-wide CPU Utilization — Training Window")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fleet_cpu_timeline.png", dpi=150)
plt.close()

# 2. CPU distribution
plt.figure(figsize=(9, 4))
plt.hist(train_cpu["avg_cpu"], bins=60)
plt.xlabel("VM-level average CPU utilization (%)")
plt.ylabel("Number of readings")
plt.title("Distribution of VM CPU Utilization")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cpu_distribution.png", dpi=150)
plt.close()

# 3. High-utilization VM percentage over time
high_cpu_percentage = (
    train_cpu.assign(high_cpu=train_cpu["avg_cpu"] >= 80)
    .groupby("timestamp")["high_cpu"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(11, 4))
plt.plot(high_cpu_percentage.index / 3600, high_cpu_percentage.values)
plt.xlabel("Elapsed time (hours)")
plt.ylabel("VM readings at 80%+ CPU (%)")
plt.title("High-Utilization VM Share")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "high_cpu_share.png", dpi=150)
plt.close()

# 4. CPU by VM category
segmented = train_cpu.merge(
    metadata[["vm_id", "vm_category"]],
    on="vm_id",
    how="left",
    validate="many_to_one",
)

category_cpu = (
    segmented.groupby("vm_category", dropna=False)["avg_cpu"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 4))
category_cpu.plot(kind="bar")
plt.xlabel("VM category")
plt.ylabel("Mean CPU utilization (%)")
plt.title("CPU Utilization by VM Category")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cpu_by_category.png", dpi=150)
plt.close()

print("\n=== TRAINING-WINDOW FINDINGS ===")
print("Mean fleet CPU:", round(fleet_cpu.mean(), 4))
print("Min fleet CPU:", round(fleet_cpu.min(), 4))
print("Max fleet CPU:", round(fleet_cpu.max(), 4))
print("Mean high-CPU VM share (%):", round(high_cpu_percentage.mean(), 4))

print("\nCPU by VM category:")
print(category_cpu)

print("\nCharts saved to:", OUTPUT_DIR)


# 5. CPU by virtual core and memory bucket
for column, filename, title in [
    ("virtual_core_bucket", "cpu_by_core_bucket.png",
     "CPU Utilization by Virtual Core Bucket"),
    ("memory_gb_bucket", "cpu_by_memory_bucket.png",
     "CPU Utilization by Memory Bucket"),
]:
    segment_data = train_cpu.merge(
        metadata[["vm_id", column]],
        on="vm_id",
        how="left",
        validate="many_to_one",
    )

    summary = (
        segment_data.groupby(column, dropna=False)["avg_cpu"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )

    print(f"\n=== {column.upper()} ===")
    print(summary)

    plt.figure(figsize=(10, 4))
    plt.bar(summary.index.astype(str), summary["mean"])
    plt.xlabel(column)
    plt.ylabel("Mean CPU utilization (%)")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()

# 6. Identify training-window peaks and low-demand periods
peak_periods = fleet_cpu.nlargest(5)
low_periods = fleet_cpu.nsmallest(5)

print("\n=== TOP 5 FLEET CPU PEAKS ===")
print(peak_periods)

print("\n=== TOP 5 LOW-DEMAND PERIODS ===")
print(low_periods)

# 7. Compare high-CPU pressure with fleet-wide demand
pressure = pd.DataFrame({
    "fleet_cpu": fleet_cpu,
    "high_cpu_vm_share_pct": high_cpu_percentage,
})

print("\n=== PEAK-RISK RELATIONSHIP ===")
print(
    "Correlation between fleet CPU and high-CPU VM share:",
    pressure["fleet_cpu"].corr(pressure["high_cpu_vm_share_pct"])
)


# 8. VM lifetime metadata and observed churn
print("\n=== VM LIFETIME METADATA ===")

print(
    metadata[
        ["timestamp_vm_created", "timestamp_vm_deleted"]
    ].describe(include="all")
)

print("\n=== OBSERVED VM CHURN ===")

observed_counts = (
    train_cpu.groupby("timestamp")["vm_id"]
    .nunique()
    .sort_index()
)

count_changes = observed_counts.diff().dropna()

print("Unique active VM counts:", observed_counts.unique())
print("Nonzero count changes:", (count_changes != 0).sum())

# Compare VM identities across consecutive timestamps.
first_timestamp = training_timestamps[0]
last_timestamp = training_timestamps[-1]

first_vms = set(
    train_cpu.loc[
        train_cpu["timestamp"] == first_timestamp, "vm_id"
    ]
)

last_vms = set(
    train_cpu.loc[
        train_cpu["timestamp"] == last_timestamp, "vm_id"
    ]
)

print("VMs entering between first and last snapshots:",
      len(last_vms - first_vms))

print("VMs leaving between first and last snapshots:",
      len(first_vms - last_vms))

