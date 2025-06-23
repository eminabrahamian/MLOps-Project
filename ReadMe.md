# Breast Cancer Classification MLOps Pipeline

[![CI](https://github.com/2025-IE-MLOps-course/mlops_project-CICD/actions/workflows/ci.yml/badge.svg)](https://github.com/2025-IE-MLOps-course/mlops_project-CICD/actions/workflows/ci.yml)

This repository implements a robust, modular MLOps pipeline for binary classification of breast cancer using the Wisconsin dataset. Designed for both academic and practical use, the project demonstrates best practices in reproducibility, configuration-driven workflows, and production-ready deployment.

---

## 🚦 Project Status

- **Modularized pipeline**: All steps (data ingestion, validation, preprocessing, training, evaluation, inference) are implemented as testable Python modules.
- **Configuration-driven**: All settings are managed via `configs/config.yaml` for easy reproducibility and experimentation.
- **Unit tested**: Extensive pytest coverage across all modules.
- **API serving**: FastAPI app exposes prediction endpoints for real-time and batch inference.
- **CI/CD**: Automated testing via GitHub Actions.
- **Artifacts**: All models, metrics, and pipelines are versioned and stored for traceability.

---

## 📁 Repository Structure

```text
.
├── configs/                  # YAML configs for pipeline and environment
├── data/                     # Raw, split, processed, and inference data
├── models/                   # Trained models, metrics, preprocessing pipelines
├── logs/                     # Log files and validation reports
├── notebooks/                # Exploratory Jupyter notebooks
├── src/                      # All pipeline source code (modularized)
│   ├── data_loader/          # Data loading utilities
│   ├── data_validator/       # Schema and data validation
│   ├── preprocessing/        # Preprocessing pipeline construction
│   ├── features/             # Feature engineering (optional)
│   ├── model/                # Model training and artifact management
│   ├── evaluation/           # Model evaluation and metrics
│   ├── inference/            # Batch inference logic
│   └── main.py               # Pipeline orchestration entry point
├── app/                      # FastAPI app for online serving
├── tests/                    # Unit tests for all modules
├── Dockerfile                # Containerization for deployment
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment definition
├── setup.sh                  # Setup script for local dev
└── README.md                 # Project documentation
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

**Run the pipeline using MLflow:**

- **Full pipeline (all steps):**
  ```bash
  mlflow run . -P steps=all
  ```
- **Run specific steps (e.g., data, train, evaluation):**
  ```bash
  mlflow run . -P steps=data,train,evaluation
  ```
- **Batch inference:**
  ```bash
  mlflow run . -P steps=infer -P input_csv=data/raw/new_inference_data.xlsx -P output_csv=data/inference_predictions/new_predictions.xlsx
  ```

All steps, parameters, and artifact paths are controlled via `configs/config.yaml`.

---

## 4. Unit Tests

Unit tests are located under `tests/`:

```
tests/
├── test_data_loader.py
├── test_data_validator.py
├── ...
```

Run all tests with coverage:

```bash
PYTHONPATH=$PWD pytest --cov=src
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

4. **"X does not have valid feature names" Warning**  
   - The code wraps the transformed NumPy array into a DataFrame with the original feature names before calling `model.predict()`. Ensure you have the latest `src/inference.py`.

5. **Excel Read/Write Errors**  
   - Ensure `openpyxl` is installed (`conda list` should show it).  
   - Verify file paths and extensions (`.csv` vs `.xlsx`).  

---

## 7. API Serving

**Serve the model via FastAPI:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- Access `/docs` for interactive API documentation.

---

## 8. Docker Deployment

**Build and run the API locally:**
```bash
docker build -t breast-cancer-api .
docker run --env-file .env -p 8000:8000 breast-cancer-api
```
- The server uses the `PORT` environment variable (default: 8000).

---

## 9. Academic Notes

- Demonstrates best practices in modularity, testing, and reproducibility.
- All logic is config-driven for easy extension and experimentation.
- Suitable for both teaching and real-world MLOps scenarios.

---

## 👩‍💻 Authors and Acknowledgments

- Developed as part of the IE University MLOps curriculum.
- Inspired by open-source MLOps projects and healthcare analytics use cases.

---

## 📜 License

This project is for academic and educational purposes.
