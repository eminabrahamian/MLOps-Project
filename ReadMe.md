# Breast Cancer Classification MLOps Pipeline

[![CI](https://github.com/2025-IE-MLOps-course/mlops_project-CICD/actions/workflows/ci.yml/badge.svg)](https://github.com/2025-IE-MLOps-course/mlops_project-CICD/actions/workflows/ci.yml)

This repository implements a robust, modular MLOps pipeline for binary classification of breast cancer using the Wisconsin Health Labs dataset. Designed for both academic and practical use, the project demonstrates MLOps best practices in reproducibility, configuration-driven workflows, and production-ready deployment.

---

## 🚦 Project Status

- **Modularized pipeline**: All stages of the ML lifecycle — including **data ingestion, validation, preprocessing, training, evaluation, and inference** — are implemented as isolated, testable modules. Each step is callable independently and orchestrated via `main.py`.

- **Configuration-driven**: The entire pipeline is **Hydra-managed** through `configs/config.yaml`, enabling easy experimentation and reproducibility with centralized control of paths, hyperparameters, seeds, and step logic.

- **Tracked and reproducible experiments**: The project uses **MLflow Projects** to orchestrate modular step-wise execution and manage environment reproducibility via `conda.yaml` and `MLproject`.

- **End-to-end experiment tracking**: **Weights & Biases (W&B)** is fully integrated to log metrics, artifacts, and configuration values across all pipeline stages, enabling robust lineage and experiment versioning.

- **Version-controlled data and models**: **DVC** is used to manage data artifacts, trained models, and intermediate outputs. It ensures that results are reproducible and versioned alongside code.

- **Unit tested**: All modules are extensively tested with `pytest`. Tests are integrated into the CI pipeline to maintain code stability.

- **API serving**: The trained model is exposed via a **FastAPI** service (`/predict` endpoint), allowing for **real-time and batch inference**. This is containerized with Docker for portability.

- **CI/CD**: A complete **GitHub Actions workflow** (`.github/workflows/ci.yml`) runs on every push. It includes:
  - Environment setup with Miniconda
  - Full test suite execution
  - MLflow pipeline run
  - W&B integration
  - Docker build checks
  - CI badge support

- **Dockerized deployment**: The FastAPI app is packaged via Docker and deployed using **Render** for cloud inference.

- **Artifacts & logging**: All trained models, preprocessing pipelines, and evaluation metrics are versioned with W&B and DVC. Logs are standardized with consistent formatting and debugging support.

---

## 📁 Repository Structure

```text
.
├── app/ # FastAPI app for serving predictions
├── configs/ # Hydra YAML configs for pipeline and environment
├── data/ # Raw, split, processed, and inference data (DVC-tracked)
├── models/ # Trained models, metrics, and preprocessing pipelines (DVC-tracked)
├── logs/ # Log files and data validation reports
├── notebooks/ # Jupyter notebooks for exploratory analysis
├── src/ # All pipeline source code (modularized steps)
│ ├── data_loader/ # Data loading logic and schema handling
│ ├── data_validator/ # Schema validation and integrity checks
│ ├── preprocessing/ # Preprocessing pipeline and transformers
│ ├── features/ # Feature engineering (optional/custom)
│ ├── model/ # Model training and artifact creation
│ ├── evaluation/ # Evaluation logic and metrics computation
│ ├── inference/ # Batch inference logic and artifact loading
│ └── main.py # Pipeline orchestrator via Hydra + MLflow
├── tests/ # Pytest-based unit tests for all pipeline modules
├── .github/ # GitHub Actions workflows (CI/CD pipelines)
├── MLproject # MLflow project file for experiment entry point
├── Dockerfile # Dockerfile for API containerization and deployment
├── render.yaml # Render-specific deployment configuration
├── main.py # Entry point script for running pipeline directly
└── README.md # Project documentation and instructions
```
---

## Problem Description

The pipeline classifies cancer diagnoses using clinical and demographic features from a breast cancer dataset. It focuses on distinguishing between **malignant** and **benign** tumors based on features derived from digitized images of fine needle aspirate (FNA) biopsies. The pipeline is designed for modular analysis, reproducible experimentation, and scalable deployment in applied machine learning settings.

### 📖 Data Dictionary

| Feature        | Description |
|----------------|-------------|
| `radius_mean`  | Mean of distances from center to points on the perimeter |
| `texture_mean` | Standard deviation of gray-scale values |
| `perimeter_mean` | Mean size of the perimeter of the cell nuclei |
| `area_mean`    | Mean size of the nuclei area |
| `smoothness_mean` | Variation in radius lengths |
| `compactness_mean` | Combination of perimeter and area |
| `concavity_mean` | Severity of concave portions of the contour |
| `symmetry_mean` | Symmetry of the cell nuclei |
| `fractal_dimension_mean` | Fractal dimension indicating complexity |

*Additional features include standard error and "worst" (largest) values for each attribute (e.g., `radius_se`, `radius_worst`, etc.). See the feature extraction code and dataset documentation for full details.*

---

## 🔁 Pipeline Overview

This MLOps pipeline is modular, reproducible, and driven entirely by configuration files. Each core component is encapsulated in its own script under `src/` and can be executed independently or via MLflow orchestration. The pipeline follows an industry-grade structure with data versioning, model tracking, and automated deployment.

### 📥 Data Loading (`data_loader.py`)
- Loads raw input data from CSV or Excel formats using a configurable path and schema.
- Supports flexible encoding and header handling.
- All parameters are defined in `configs/config.yaml`, and logging is handled centrally.

### ✅ Data Validation (`data_validator.py`)
- Validates the raw data against schema rules defined in `config.yaml`, including:
  - Required columns and data types
  - Value ranges and allowed sets
  - Missing values
- Generates a JSON report and logs all warnings or errors.

### 🧹 Preprocessing (`preprocessing.py`)
- Builds a **leakage-proof**, sklearn-compatible pipeline using:
  - Imputation
  - Scaling
  - Optional BMI, risk score, datetime, outlier, and interaction features
  - Categorical encoding (OneHot or Ordinal)
- Transforms data consistently across train/valid/test/inference.
- Saves processed data to `data/processed/`.

### 🧠 Model Training (`model.py`)
- Trains a `KNeighborsClassifier` using the preprocessed data.
- Saves the trained model and pipeline using `pickle`.
- Supports hyperparameter tuning via `config.yaml`.
- Automatically logs validation and test set metrics using the evaluation module.

### 📊 Evaluation (`evaluation.py`)
- Computes configurable metrics such as accuracy, F1-score, specificity, and ROC AUC.
- Outputs results as both logs and JSON files for easy tracking.
- Compatible with batch evaluation or in-pipeline integration.

### 📈 Feature Engineering (`features.py`)
- Includes optional transformers for:
  - Risk score aggregation
  - BMI calculation
  - Interaction features
  - Date/time expansion
  - Outlier flagging
- All transformers are registered and modular.
- These are only applied if enabled via `config.yaml`.

### 🔮 Inference (`inference.py`)
- Loads raw inference data, applies the saved preprocessing pipeline and trained model.
- Generates predictions (optionally with class probabilities).
- Outputs results to Excel and logs prediction stats.

Each of these components is tested with `pytest`, tracked with Weights & Biases, and reproducible via MLflow.
---

## ⚙️ Configuration and Reproducibility

- **`config.yaml`**: Central control hub for the pipeline — defines paths, model type (`knn`), feature specifications, logging setup, evaluation metrics, and active pipeline steps. Enables modular, reproducible execution through Hydra.

- **`environment.yml`**: Defines a consistent, Conda-based Python environment across local and remote runs. Includes dependencies such as:
  - `scikit-learn`, `pandas`, `numpy`, `matplotlib`
  - `pyyaml`, `openpyxl`, `mlflow`, `wandb`
  - Ensures all steps run identically in CI/CD and cloud deployment.

- **Artifacts & State Tracking**:
  - Preprocessing pipelines, trained models, raw/processed data splits, and metric reports are automatically versioned and saved to structured directories.
  - Integration with **DVC** allows tracking of data and model files.
  - W&B and MLflow track model runs, configuration lineage, and evaluation outputs for full experiment traceability.

---

## 🚀 Quickstart

**📦 Environment setup**
```bash
conda env create -f environment.yml
conda activate mlops_project
./setup.sh            # Installs Python dependencies and sets PYTHONPATH
dvc pull              # Pull tracked data and models
wandb login           # Authenticate with Weights & Biases
cp .env.example .env  # Create local environment file
# Then edit `.env` to include:
# WANDB_PROJECT=<your-project-name>
# WANDB_ENTITY=<your-wandb-entity>
```

**▶️ Run the full pipeline**
```bash
# Option 1: using the main orchestrator with Hydra
python main.py main.steps=all

# Option 2: using MLflow (configured via MLproject)
mlflow run . -P steps=all

# Override model hyperparameters (e.g., n_neighbors)
python main.py main.steps=model main.hydra_options="model.knn.params.n_neighbors=3"
mlflow run . -P steps=model -P hydra_options="model.knn.params.n_neighbors=3"
```

> `main.steps` can be any subset of: `data_loader,data_validator,preprocessing,model,evaluation,inference`

**🧪 Run tests**
```bash
PYTHONPATH=$PWD pytest --cov=src
```

_(Note: your tests must be invoked from the root folder using the path `tests/`)_

---

**🔍 Run standalone inference**
```bash
python -m src.inference.inference \
  data/inference/new_data.xlsx \
  configs/config.yaml \
  data/inference/output_predictions.xlsx --proba
```

---

**🌐 Serve the model via FastAPI**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**📬 Call the running API (batch)**
```bash
python scripts/call_api.py \
  --url http://localhost:8000/predict_batch \
  --input data/inference/new_data.xlsx
```

> The `/predict` endpoint supports single-record requests. See the Swagger UI at `http://localhost:8000/docs`.

---

## 🐳 Docker Deployment

Before building the image, ensure the following environment variables are set in your `.env`:

```env
WANDB_PROJECT=<your-project>
WANDB_ENTITY=<your-entity>
WANDB_API_KEY=<your-api-key>
```

**Build and run Docker container locally**
```bash
docker build -t cancer-knn-api .
docker run --env-file .env -p 8000:8000 cancer-knn-api
```

Render or other cloud platforms will respect the `PORT` variable (default `8000`).  
See `render.yaml` for a minimal, production-ready deployment configuration.

---

## 9. Academic Notes

- Demonstrates best practices in modularity, testing, and reproducibility using MLFlow and W&B.
- All logic is config-driven and hydra-managed for easy extension and experimentation.
- Suitable for both teaching and real-world MLOps scenarios.

---

## 👩‍💻 Authors and Acknowledgments

- Developed as part of the IE University MLOps curriculum.
- Inspired by open-source MLOps projects and healthcare analytics use cases.

---

## 📜 License

This project is for academic and educational purposes. See the attached license for more details.
