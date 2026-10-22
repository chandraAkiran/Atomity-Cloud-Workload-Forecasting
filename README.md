# Cloud Workload Forecasting & Analysis

**AI/ML Internship Assignment | Atomity**

A reproducible machine learning project for forecasting cloud infrastructure workload demand using the Microsoft Azure Public Dataset V2 curated VM trace.

The project analyzes virtual machine CPU utilization, compares forecasting methods using chronological validation, and predicts cloud workload demand for the next **2 hours (24 five-minute intervals)**.

## 1. Project Overview

Cloud infrastructure providers must forecast workload demand to make effective resource allocation and capacity planning decisions.

Underestimating demand can lead to performance degradation and service interruptions, while overestimating demand can result in unnecessary infrastructure costs.

This project addresses that challenge by forecasting two metrics:

| Forecast target | Description |
|---|---|
| `active_vm_count` | Number of unique virtual machines reporting valid CPU observations at each timestamp |
| `aggregate_cpu_utilization` | Mean VM-level average CPU utilization, expressed as a percentage |

**Project objectives:**

- Explore cloud workload patterns and CPU utilization behavior.
- Analyze workload differences across VM categories and resource configurations.
- Develop and evaluate time-series forecasting baselines.
- Compare a machine learning approach against simpler forecasting methods.
- Generate 24 future workload predictions.
- Recommend practical cloud capacity planning strategies.

## 2. Dataset Description

**Source:** Microsoft Azure Public Dataset V2, distributed through the official Atomity forecasting challenge.

| Dataset characteristic | Value |
|---|---:|
| Virtual machines | 10,000 |
| Total CPU observations | 1,340,000 |
| Unique timestamps | 134 |
| Sampling interval | 300 seconds (5 minutes) |
| Training timestamps | 110 |
| Holdout timestamps | 24 |
| Forecast horizon | 2 hours |

### Dataset Files

**`vm_cpu_readings.csv.gz`**

Contains timestamp-level VM CPU measurements, including:

- `timestamp`
- `vm_id`
- `min_cpu`
- `max_cpu`
- `avg_cpu`

**`vm_metadata.csv.gz`**

Contains VM configuration and workload information, including VM category, virtual core bucket, memory bucket, and VM lifecycle metadata.

### Chronological Data Split

To prevent data leakage, the observations are divided chronologically.

| Partition | Timestamps | Observations |
|---|---|---:|
| Training | 0–32700 | 110 |
| Final holdout | 33000–39900 | 24 |

The final 24 timestamps are excluded from model training, feature selection, hyperparameter selection, and model comparison.

## 3. Project Structure

```text
Atomity-Cloud-Workload-Forecasting/
│
├── challenge_data/
│   ├── vm_cpu_readings.csv.gz
│   ├── vm_metadata.csv.gz
│   ├── DATA_CARD.md
│   └── SHA256SUMS
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── explore_workload.py
│   ├── evaluate_baselines.py
│   ├── evaluate_ridge.py
│   ├── generate_predictions.py
│   ├── predict_from_data.py
│   ├── evaluate_holdout.py
│   └── run_pipeline.py
│
├── tests/
│   ├── test_generate_predictions.py
│   └── test_predict_from_data.py
│
├── reports/
│   ├── figures/
│   └── final_report.md
│
├── predictions.csv
├── pyproject.toml
├── README.md
├── ASSIGNMENT.md
├── DATASET.md
└── SUBMISSION.md
```

**Note:** The source dataset and generated intermediate files are excluded from Git where appropriate. The official dataset must be downloaded separately.

## 4. Installation and Setup

### Prerequisites

- Python 3.11 or newer
- Git
- Internet connection for downloading the official dataset

### Step 1 — Clone the Repository

```bash
git clone https://github.com/chandraAkiran/Atomity-Cloud-Workload-Forecasting.git

cd Atomity-Cloud-Workload-Forecasting
```

### Step 2 — Create a Virtual Environment

On macOS or Linux:

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
python -m pip install -e ".[dev]"
```

The project uses Python, NumPy, pandas, Matplotlib, scikit-learn, and pytest.

## 5. Download the Official Dataset

Download the official dataset from:

[Atomity Forecasting Challenge — Data Release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)

Extract the files into the `challenge_data/` directory.

Expected structure:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

### Verify Dataset Integrity

On macOS:

```bash
cd challenge_data

shasum -a 256 -c SHA256SUMS

cd ..
```

Expected output:

```text
vm_cpu_readings.csv.gz: OK
vm_metadata.csv.gz: OK
DATA_CARD.md: OK
```

## 6. Reproduce the Forecasts

### 6.1 Generate the Final Predictions — Recommended

The primary prediction command is:

```bash
python -m scripts.predict_from_data \
  --data challenge_data \
  --output predictions.csv
```

This command:

1. Loads `vm_cpu_readings.csv.gz` from the specified dataset directory.
2. Aggregates VM CPU observations into timestamp-level forecasting targets.
3. Reserves the final 24 timestamps as the holdout period.
4. Uses only earlier observations to generate predictions.
5. Applies the selected forecasting models:
   - **Active VM count:** Last-value persistence.
   - **Aggregate CPU utilization:** 12-step moving average.
6. Saves the final 24 predictions to `predictions.csv`.

The command supports datasets with different VM populations and history lengths, provided they contain at least 36 consecutive five-minute timestamps.

This allows reviewers to test the forecasting workflow using smaller datasets without requiring the exact 10,000-VM, 134-timestamp public release.

Expected output on the official dataset:

```text
Training observations: 110
Forecast rows: 24
Saved predictions to: predictions.csv
```

### 6.2 Prediction File Format

The submission file contains exactly 24 rows and three columns:

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

The output must satisfy the official submission contract:

- Exactly 24 forecast rows.
- Chronologically ordered timestamps.
- No duplicate timestamps.
- No missing or non-finite predictions.
- Nonnegative active VM counts.
- CPU utilization values between 0% and 100%.
- No extra index column.

### 6.3 Run the Complete Analysis Pipeline

To reproduce the complete analysis and model development workflow on the official dataset, run:

```bash
python scripts/run_pipeline.py
```

This command executes:

1. **Dataset inspection:** Loads and validates the Azure VM data and constructs timestamp-level forecasting targets.
2. **Exploratory data analysis:** Investigates CPU behavior, high-utilization workloads, VM categories, and resource buckets.
3. **Baseline validation:** Evaluates persistence, moving average, and linear trend methods using chronological folds.
4. **Machine learning comparison:** Evaluates direct multi-output Ridge regression against the selected baseline.
5. **Final prediction generation:** Generates the required 24-step forecasts using training observations only.

The full pipeline produces analysis figures and evaluation metrics under the `reports/` directory.

**Important:** `run_pipeline.py` is designed for the official dataset and its predefined validation folds. For small-dataset compatibility testing, use the recommended `predict_from_data` command.

## 7. Exploratory Data Analysis

Exploratory analysis is restricted to the training window to avoid using future observations during model development.

### Key Findings

- The curated trace contains 10,000 active VMs at every training timestamp.
- Average fleet CPU utilization is approximately **9.41%** during training.
- Approximately **1.27% of VMs** have CPU utilization at or above 80% per training timestamp.
- Interactive workloads show higher average CPU utilization than delay-insensitive workloads.
- VM CPU utilization differs across virtual core and memory buckets.
- The observed training window contains no active-VM-count variation.

These findings suggest that fleet-average CPU utilization alone is insufficient for identifying localized workload pressure.

### Generated Visualizations

The exploratory analysis generates figures in `reports/figures/`, including:

- `fleet_cpu_timeline.png`
- `cpu_distribution.png`
- `high_cpu_share.png`
- `cpu_by_category.png`
- `cpu_by_core_bucket.png`
- `cpu_by_memory_bucket.png`

## 8. Forecasting Models

Four forecasting approaches were compared.

### Model 1 — Persistence

Predicts that the future value will equal the most recent observed value.

Used as the selected forecasting method for `active_vm_count`.

### Model 2 — 12-Step Moving Average

Predicts future CPU utilization using the mean of the previous 12 observations.

Because each observation represents five minutes, this model uses the most recent **60 minutes** of CPU history.

Selected as the final aggregate CPU forecasting model.

### Model 3 — Linear Trend

Fits a linear trend using the most recent 24 observations and extrapolates into the forecast horizon.

### Model 4 — Ridge Regression

Uses a regularized linear regression model with lag-based and summary features to generate a direct 24-step forecast.

The Ridge model was evaluated as a machine learning alternative but did not outperform the moving-average baseline.

## 9. Temporal Validation Strategy

Random train-test splitting is inappropriate for this forecasting task because it can expose future information to the model.

Instead, the project uses expanding-window chronological validation within the first 110 observations.

| Fold | Training observations | Validation observations | Forecast horizon |
|---|---:|---:|---:|
| 1 | 38 | 24 | 24 |
| 2 | 62 | 24 | 24 |
| 3 | 86 | 24 | 24 |

All models are evaluated on comparable forecast origins and validation periods.

### Evaluation Metrics

**Mean Absolute Error (MAE)** measures average absolute forecast deviation.

**Root Mean Squared Error (RMSE)** penalizes larger forecasting errors more heavily.

**Normalized MAE** expresses forecast error relative to average CPU utilization.

Additional analysis examines prediction errors during high-demand periods, using peak thresholds derived from each fold's training history.

## 10. Model Comparison Results

### Aggregate CPU Utilization — Training-Only Validation

| Model | Mean MAE (percentage points) | Mean RMSE (percentage points) |
|---|---:|---:|
| Persistence | 0.2082 | 0.2566 |
| **12-step moving average** | **0.1784** | **0.2243** |
| Linear trend | 0.3656 | 0.4336 |
| Ridge regression | 0.3778 | 0.4257 |

The 12-step moving average achieved approximately **14.32% lower MAE** than persistence during temporal validation.

Although Ridge regression was more complex, it performed worse on the available short training history.

### Active VM Count

Persistence produced zero validation error because the active VM population remained constant at 10,000 throughout the training period.

This result reflects the stability of the curated dataset and should not be interpreted as evidence of performance during VM creation or termination events.

## 11. Final Forecasts

The selected models generate 24 forecasts for the final two-hour horizon.

| Forecast target | Predicted value |
|---|---:|
| Active VM count | 10,000 |
| Aggregate CPU utilization | Approximately 9.21637% |
| Forecast timestamps | 33000–39900 |
| Forecast interval | 300 seconds |
| Number of forecast rows | 24 |

The predictions are stored in:

`predictions.csv`

## 12. Final Holdout Evaluation

After the model selection and prediction file were finalized, the last 24 observed timestamps were used for a separate holdout evaluation.

Run:

```bash
python scripts/evaluate_holdout.py
```

This evaluation reads the frozen predictions and the observed holdout targets. It does not modify `predictions.csv`.

### Holdout Results

| Metric | Active VM Count | Aggregate CPU Utilization |
|---|---:|---:|
| MAE | 0 | 0.1230 percentage points |
| RMSE | 0 | 0.1613 percentage points |
| Normalized MAE | 0% | 1.327% |

The aggregate CPU forecast had a mean bias of approximately **-0.0526 percentage points**, indicating slight underprediction on this holdout period.

No holdout CPU observations exceeded the peak threshold derived from the training data, so holdout peak-demand forecasting performance could not be meaningfully assessed.

The holdout evaluation is reported for transparency and was not used to choose the forecasting model.

## 13. Automated Testing

The project includes tests for:

- Forecast schema and 24-step horizon.
- Valid forecast values and timestamp cadence.
- Independence from older observations outside the selected moving-average window.
- Rejection of insufficient training history.
- Support for variable-length training histories.
- End-to-end prediction generation from a smaller synthetic VM dataset.

Run:

```bash
python -m pytest -q
```

Expected result:

```text
8 passed
```

The synthetic smoke test uses only three VMs and 40 timestamps, demonstrating that the prediction entrypoint does not depend on the official dataset's exact VM population or timestamp count.

## 14. Cloud Capacity Planning Recommendations

Forecasting results should support infrastructure monitoring and capacity planning rather than trigger automatic scaling decisions without additional evidence.

### Recommendations

1. **Maintain sufficient VM capacity:** The observed workload consistently contains 10,000 active VMs, so the forecast provides no evidence of near-term VM-count reduction.

2. **Monitor workload hotspots:** Approximately 1.27% of VMs reach high CPU utilization despite low average fleet CPU usage.

3. **Track multiple operational signals:** Combine fleet-average CPU forecasts with high-utilization VM share, workload categories, resource buckets, and other infrastructure metrics.

4. **Maintain operational headroom:** Use a conservative monitoring buffer to account for forecast uncertainty and unexpected workload changes.

5. **Avoid aggressive downsizing:** Low average CPU utilization does not necessarily imply that all workloads can safely operate with fewer resources.

An initial monitoring heuristic can be calculated as:

**Forecast CPU + 2 × validation RMSE**

Using the selected model:

9.21637% + 2 × 0.22430% ≈ **9.665%**

This is an illustrative monitoring threshold, not a calibrated prediction interval or automatic scaling rule.

## 15. Limitations

The project has several important limitations:

- The dataset covers a short observation period.
- The active VM population is constant within the observed window.
- The models are not validated on sudden large-scale workload changes.
- The aggregate CPU target is an unweighted average across VMs.
- Fleet-average CPU does not directly represent physical CPU core consumption.
- Only a single 24-step holdout window is available for final evaluation.
- The Ridge regression model has limited historical examples for learning longer-range patterns.
- The evaluation does not establish reliable prediction intervals.

### Future Improvements

Potential improvements include:

- Testing on longer cloud workload histories.
- Modeling VM arrivals and departures.
- Forecasting workload separately by VM category.
- Incorporating additional infrastructure signals.
- Evaluating uncertainty-aware forecasting methods.
- Testing adaptive forecasting models under workload drift.

## 16. Reports and Deliverables

| File or directory | Purpose |
|---|---|
| `predictions.csv` | Required 24-step forecast submission |
| `reports/final_report.md` | Technical report, findings, evaluation, and recommendations |
| `reports/figures/` | Exploratory analysis visualizations |
| `scripts/predict_from_data.py` | Flexible prediction entrypoint |
| `scripts/run_pipeline.py` | Complete analysis and forecasting pipeline |
| `scripts/evaluate_holdout.py` | Final frozen holdout evaluation |
| `tests/` | Automated validation and smoke tests |
| `pyproject.toml` | Dependency and project configuration |

## 17. Reproducibility and Data Leakage Prevention

The project follows these principles:

- Chronological separation of training and holdout observations.
- Model comparison performed using training-only validation.
- Feature construction restricted to observations available at each forecast origin.
- Final predictions generated without fitting on holdout target values.
- Holdout evaluation performed only after the predictions are generated.
- Automated tests for output correctness and smaller dataset compatibility.
- Explicit dependency declaration and documented execution commands.
- Source datasets excluded from version control.

## 18. References

- [Atomity Forecasting Challenge](https://github.com/atomityhq/forecasting-challenge)
- [Official Assignment](ASSIGNMENT.md)
- [Dataset Contract](DATASET.md)
- [Submission Requirements](SUBMISSION.md)
- [Microsoft Azure Public Dataset](https://github.com/Azure/AzurePublicDataset)
- [Official Challenge Dataset Release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)

---

**Repository:** [Atomity Cloud Workload Forecasting](https://github.com/chandraAkiran/Atomity-Cloud-Workload-Forecasting)

**Project status:** Forecasting pipeline implemented, 24-step predictions generated, temporal validation completed, holdout results documented, and automated tests passing.
