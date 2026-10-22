
# Cloud Workload Forecasting & Analysis

AI/ML take-home assignment for Atomity using the Microsoft Azure
Public Dataset V2 curated VM trace.

## Project overview

This project forecasts cloud workload demand over a two-hour horizon
using five-minute VM CPU observations.

Forecast targets:
- `active_vm_count`: number of distinct VMs reporting CPU data.
- `aggregate_cpu_utilization`: mean VM-level average CPU utilization (%).

The dataset contains 10,000 VMs, 134 timestamps, and 1,340,000
CPU observations.

The first 110 timestamps are used for model development.
The final 24 timestamps are reserved for holdout evaluation.

## Key results

| Metric | Result |
|---|---:|
| Selected CPU model | 12-step moving average |
| Active VM model | Persistence |
| CPU validation MAE | 0.1784 percentage points |
| CPU validation RMSE | 0.2243 percentage points |
| Improvement over persistence MAE | 14.32% |
| Final holdout CPU MAE | 0.1230 percentage points |
| Final holdout CPU RMSE | 0.1613 percentage points |
| Final holdout normalized CPU MAE | 1.327% |
| Final holdout VM-count MAE | 0 |

The selected CPU forecast is 9.21637%, with 10,000 active VMs
at each of the 24 future timestamps.

## Installation

Requirements: Python 3.11 or newer.

```bash
git clone https://github.com/chandraAkiran/Atomity-Cloud-Workload-Forecasting.git
cd Atomity-Cloud-Workload-Forecasting

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -e ".[dev]"
```

## Download the dataset

Download the official `data-v1` release:

https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1

Extract the supplied files into `challenge_data/`.

Expected files:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

On macOS, verify the extracted checksums:

```bash
cd challenge_data
shasum -a 256 -c SHA256SUMS
cd ..
```

The large source dataset is intentionally excluded from Git.

## Reproduce the forecasts

Run the full training and forecast-generation workflow:

```bash
python scripts/run_pipeline.py
```

This command:
1. Inspects the dataset and constructs timestamp-level targets.
2. Runs training-only exploratory analysis.
3. Evaluates temporal forecasting baselines.
4. Evaluates the Ridge regression alternative.
5. Generates `predictions.csv` from the first 110 timestamps.

The holdout labels are not used for selecting or generating forecasts.

The generated file contains exactly 24 rows with columns:

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

## Test the project

```bash
python -m pytest -q
```

## Evaluate the frozen holdout predictions

After model selection and prediction generation:

```bash
python scripts/evaluate_holdout.py
```

This reads the final 24 actual target observations for evaluation
and does not modify `predictions.csv`.

## Modeling and validation

Three expanding-window folds evaluate the same 24-step horizon:

| Fold | Training observations | Validation observations |
|---|---:|---:|
| 1 | 38 | 24 |
| 2 | 62 | 24 |
| 3 | 86 | 24 |

Compared approaches:
- Last-value persistence
- 12-step moving average
- 24-step linear trend
- Direct multi-output Ridge regression

The 12-step moving average was selected for CPU forecasting based
on training-only validation.

VM-count persistence was selected because the curated population
remains at 10,000 VMs throughout the observed training window.

## Operational recommendation

Do not automatically remove cloud capacity based only on low average
CPU utilization.

Monitor:
- Forecasted fleet-average CPU
- VM-level high-utilization share
- Workload categories and resource buckets
- Deviations from forecast and operational headroom

A threshold near forecast plus twice the validation RMSE can be used
as an initial monitoring heuristic, not as a calibrated uncertainty
interval or automatic scaling policy.

## Reports and outputs

- `reports/final_report.md`: technical report and recommendations
- `reports/figures/`: training-window visualizations
- `predictions.csv`: required 24-step forecast
- `scripts/`: data preparation, modeling, and evaluation scripts
- `tests/`: automated tests

## Limitations

The trace covers a short observation window and has no observed
VM-count churn. Forecast results should not be generalized to
longer horizons or different infrastructure without further testing.

The aggregate CPU target is unweighted across VMs and is not a
direct measurement of total physical core utilization.

## References

- [Atomity assignment](ASSIGNMENT.md)
- [Dataset contract](DATASET.md)
- [Submission requirements](SUBMISSION.md)
- [Microsoft Azure Public Dataset](https://github.com/Azure/AzurePublicDataset)

