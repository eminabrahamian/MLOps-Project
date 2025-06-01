# Breast Cancer Classification MLOps Pipeline

This repository implements a complete, end-to-end MLOps pipeline for binary classification on the Wisconsin Breast Cancer dataset (or any similar tabular dataset). It is fully driven by configuration files (`configs/config.yaml` and `environment.yml`) and orchestrated by the main script (`src/main.py`). The pipeline covers:

1. **Data Ingestion & Validation**
2. **Preprocessing & Feature Engineering**
3. **Model Training & Evaluation**
4. **Batch Inference**
5. **Unit Testing & Linting**

All code lives under `src/`, and test cases are under `tests/`. The environment is reproducible via `environment.yml`.  

---

## Repository Layout

```
MLOps/
├── configs/
│   ├── config.yaml            ← Pipeline settings (paths, schema, hyperparameters, etc.)
│   └── environment.yml        ← Conda environment definition
│
├── data/
│   ├── raw/
│   │   ├── cancer.xlsx        ← Original dataset
│   │   └── new_inference_data.xlsx  ← Synthetic data for inference
│   │
│   ├── splits/                ← Raw train/valid/test splits (generated)
│   │   ├── train_raw.csv
│   │   ├── valid_raw.csv
│   │   └── test_raw.csv
│   │
│   ├── processed/             ← Processed train/valid/test CSVs (generated)
│   │   ├── train_processed.csv
│   │   ├── valid_processed.csv
│   │   └── test_processed.csv
│   │
│   └── inference_predictions/ ← Predictions from inference (generated)
│       └── new_predictions.xlsx
│
├── logs/
│   ├── main.log               ← Central pipeline log (generated)
│   └── validation_report.json ← Data validation report (generated)
│
├── models/
│   ├── preprocessing_pipeline.pkl  ← Serialized sklearn pipeline (generated)
│   ├── active_model.pkl            ← Serialized trained KNN model (generated)
│   └── metrics/
│       ├── validation_metrics.json ← Validation split metrics (generated)
│       ├── test_metrics.json       ← Test split metrics (generated)
│       └── combined_metrics.json   ← Combined metrics report (generated)
│
├── notebooks/
│   └── Manual & scikit KNN on Cancer Data.ipynb  ← Exploratory notebook (optional)
│
├── src/
│   ├── data/
│   │   ├── data_loader.py     ← Loads CSV/Excel into pandas DataFrame
│   │   ├── data_validator.py  ← Validates DataFrame columns against schema
│   │   └── preprocessing.py   ← Builds sklearn Pipeline from config
│   │
│   ├── features/
│   │   └── features.py        ← Optional feature-engineering transformers
│   │
│   ├── evaluation/
│   │   └── evaluation.py      ← Computes & saves classification metrics
│   │
│   ├── inference/
│   │   └── inference.py       ← Loads pipeline + model, scores new data
│   │
│   ├── models/
│   │   └── model.py           ← Splits data, trains KNN, evaluates & saves artifacts
│   │
│   ├── main.py                ← Orchestrates “data”, “train”, and “infer” stages
│   └── __init__.py
│
├── tests/
│   ├── test_data_loader.py    ← Unit tests for data_loader.py
│   └── test_data_validator.py ← Unit tests for data_validator.py
│
└── README.md                  ← This file
```

---

## 1. Environment Setup

Create a reproducible Python environment using Conda:

```bash
cd MLOps
conda env create -f configs/environment.yml
conda activate mlops_project
```

Verify that:

- Python version is 3.10.x  
- Key packages installed include `numpy`, `pandas`, `scikit-learn`, `pyyaml`, `openpyxl`, `sqlalchemy`, `requests`, `pytest`, `pytest-cov`, and `flake8`.

---

## 2. Configuration File (`configs/config.yaml`)

Edit `configs/config.yaml` to specify:

- **Data Ingestion** (`data_source`): Path to raw data, file type (`csv` or `excel`), sheet name (if Excel), header row, and encoding.  
- **Logging** (`logging`): Level, log file path, format, and datefmt.  
- **Data Splitting** (`data_split`): `test_size`, `valid_size`, `random_state`, `stratify`.  
- **Data Validation** (`data_validation`): Enable/disable, `action_on_error` ("raise" or "warn"), `report_path`, and per-column schema (`columns`).  
- **Preprocessing** (`preprocessing`): Options for renaming columns, BMI computation, ICD-10 flags, interaction features, outlier flags, datetime extraction, etc.  
- **Features** (`features`): Lists of `continuous` and `categorical` columns; `feature_columns` used by the pipeline; `raw_features` used for splitting.  
- **Model** (`model`): `active` (which algorithm, e.g., "knn"), hyperparameters under `knn.params`, and `save_path` for the trained model.  
- **Artifacts** (`artifacts`): Paths for `splits_dir`, `processed_dir`, `preprocessing_pipeline`, `model_path`, `metrics_dir`, `metrics_path`.  
- **Evaluation Metrics** (`metrics`): List of metric names (e.g., `"accuracy"`, `"precision"`, `"recall"`, `"f1"`, `"roc auc"`, `"specificity"`, `"npv"`, `"confusion matrix"`).  
- **Inference** (`inference`): `return_proba` (true/false).

Make sure **all** feature lists match the column names exactly (case- and space-sensitive) and that `raw_features` and `features.feature_columns` exclude `id` and `target`.

---

## 3. How to Run the Pipeline

### A. Data Ingestion & Validation Only

```bash
python -m src.main   --config configs/config.yaml   --stage data
```

- Reads raw data from `data_source.path` (Excel or CSV).  
- Validates against `data_validation.schema` and writes `validation_report.json`.  

### B. Full Training Pipeline

```bash
python -m src.main   --config configs/config.yaml   --stage all
```

Equivalent to running both `--stage data` and `--stage train`:

1. **Data Stage**  
   - Loads and validates raw data.  

2. **Training Stage**  
   - Splits into train/valid/test using `raw_features` and `target`.  
   - Builds & fits preprocessing pipeline **on train only**.  
   - Transforms splits, reattaches `target`, and writes processed CSVs to `data/processed/`.  
   - Saves pipeline to `models/preprocessing_pipeline.pkl`.  
   - Trains a KNN (`n_neighbors`, `weights`, `metric` from config) on processed train set.  
   - Saves trained model to `models/active_model.pkl`.  
   - Evaluates on validation & test using `evaluate_classification`, writes JSON metrics under `models/metrics/`.  

After completion, you should see:

- `data/splits/{train_raw.csv, valid_raw.csv, test_raw.csv}`  
- `data/processed/{train_processed.csv, valid_processed.csv, test_processed.csv}`  
- `models/preprocessing_pipeline.pkl`  
- `models/active_model.pkl`  
- `models/metrics/{validation_metrics.json, test_metrics.json, combined_metrics.json}`  
- `logs/main.log` and `logs/validation_report.json`

### C. Batch Inference

> **Prerequisite**: You must have a trained model and pipeline (`models/preprocessing_pipeline.pkl` and `models/active_model.pkl`).

```bash
python -m src.main   --config configs/config.yaml   --stage infer   --input_csv data/raw/new_inference_data.xlsx   --output_csv data/inference_predictions/new_predictions.xlsx
```

- Reads new data (`.xlsx` or `.csv`), validates it.  
- Loads pickled pipeline & model, applies pipeline to new DataFrame (wraps results to DataFrame so KNN sees correct feature names).  
- Predicts (and probability if `return_proba: true`).  
- Writes predictions (and class probabilities) to `data/inference_predictions/new_predictions.xlsx`.  
- Logs completion in `logs/main.log`.

---

## 4. Unit Tests

Unit tests are located under `tests/`:

```
tests/
├── test_data_loader.py
└── test_data_validator.py
```

Run all tests with coverage:

```bash
pytest --cov=src
```

Aim for **>90% coverage** across data loading, validation, preprocessing, model training, evaluation, and inference.

---

## 5. Linting

Use `flake8` to catch style issues:

```bash
flake8 src
```

Ensure code conforms to PEP8 and avoids unused imports or undefined variables.

---

## 6. Troubleshooting

1. **ModuleNotFoundError**  
   - Ensure the conda environment `mlops_project` is active and contains all required packages (`numpy`, `pandas`, `scikit-learn`, `pyyaml`, `openpyxl`, `sqlalchemy`, `requests`, etc.).

2. **YAML Parsing Errors**  
   - Remove any literal `...` placeholders. YAML lists must be fully specified.  
   - Check indentation—use two spaces per level (no tabs).

3. **Missing Column Errors**  
   - If inference complains `"columns are missing: {...}"`, verify that `features.feature_columns` in `config.yaml` exactly matches the columns in your input file (`new_inference_data.xlsx`).

4. **“X does not have valid feature names” Warning**  
   - The code wraps the transformed NumPy array into a DataFrame with the original feature names before calling `model.predict()`. Ensure you have the latest `src/inference.py`.

5. **Excel Read/Write Errors**  
   - Ensure `openpyxl` is installed (`conda list` should show it).  
   - Verify file paths and extensions (`.csv` vs `.xlsx`).  

---

## 7. Contributing

- To add new models (e.g., Random Forest), update `model.active` and `model.<algorithm>.params` in `config.yaml`, then extend `src/model.py` to handle the new algorithm.  
- To add new feature transformers, implement them in `src/features.py` and enable them under `preprocessing` in the config.  
- For any bugs or improvements, please open an issue or submit a pull request.

---

With this README, a new user can clone the repo, create the environment, tweak configuration, and run the entire pipeline from raw data to batch inference. Good luck!

