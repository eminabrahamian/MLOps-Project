"""
Hydra‐driven data‐loading step.

Robust try/except, structured logging,
and optional artifact tracking via Weights & Biases.

This script ties together:
- Configuration loading (`load_config`)
- Logger setup (`setup_logger`)
- Data ingestion (`load_data`)
"""

import sys
from datetime import datetime
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

import wandb
from data_loader import DataLoaderError, load_data, setup_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@hydra.main(
    config_path="../../configs", config_name="config", version_base=None
)
def main(cfg: DictConfig) -> None:
    """
    Entry point to load the data.

    Expects in configs/config.yaml:
      data_source, logging, data_load (options),
      wandb (project/entity), env_file
    """
    # 1) Set up structured logging
    logger = setup_logger(cfg.logging)
    logger.info("Starting data_loader step")

    # Output directory and data file are all resolved from repo root
    output_dir = PROJECT_ROOT / cfg.data_load.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path_cfg = Path(cfg.data_source.raw_path)
    resolved_raw_path = (PROJECT_ROOT / raw_path_cfg).resolve()

    if not resolved_raw_path.is_file():
        raise FileNotFoundError(f"Data file not found: {resolved_raw_path}")

    # 2) Load environment variables (if they exist)
    env_path = cfg.get("env_file", ".env")
    env_file = Path(env_path)

    if env_file.is_file():
        load_dotenv(env_path)
        logger.info("Loaded environment variables from %s", env_path)
    else:
        logger.warning("No .env file found at %s — skipping load", env_path)

    run = None
    try:
        # 3) Initialize W&B run
        run = wandb.init(
            project=cfg.main.wandb.project,
            entity=cfg.main.wandb.entity,
            name=f"data_loader_{datetime.now():%Y%m%d_%H%M%S}_\
                {resolved_raw_path.name}",
            config=dict(cfg),
            job_type="data_load",
            tags=["data_loader", resolved_raw_path.name],
        )
        logger.info(
            "Initialized W&B run: %s/%s", cfg.main.wandb.project, run.name
        )

        # 4) Load data using your data_loader pipeline
        df = load_data()
        if df.empty:
            raise DataLoaderError(
                "Loaded DataFrame is empty." " Check your data source."
            )
        n_rows, n_cols = df.shape
        logger.info(
            "Data loaded successfully: %d rows, %d cols", n_rows, n_cols
        )

        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            logger.warning(
                "%d duplicates found in data. Consider removing "
                "them before moving forward.",
                duplicate_count,
            )

        # 5) Log basic metrics
        if run:
            wandb.log(
                {"n_rows": n_rows, "n_cols": n_cols, "shape": list(df.shape)}
            )

        # 6) Optionally log a sample of the data
        if cfg.data_load.get("log_sample", True) and run:
            sample_n = cfg.data_load.get("sample_n", 100)
            sample_tbl = wandb.Table(dataframe=df.head(sample_n))
            wandb.log({"sample_data": sample_tbl})
            logger.info("Logged sample of %d rows to W&B", sample_n)

        # 7) Optionally save raw data as an artifact
        if cfg.data_load.get("log_artifact", True) and run:
            artifact = wandb.Artifact("raw_data", type="dataset")
            artifact.add_file(resolved_raw_path, name=resolved_raw_path.name)
            run.log_artifact(artifact, aliases=["latest"])
            logger.info("Logged raw data artifact: %s", resolved_raw_path.name)

        wandb.summary.update(
            {
                "n_rows": n_rows,
                "n_cols": n_cols,
                "n_duplicates": duplicate_count,
                "columns": list(df.columns),
            }
        )

    except DataLoaderError as e:
        logger.error("DataLoaderError: %s", e)
        if run:
            run.alert(title="Data Load Failed", text=str(e))
        sys.exit(1)

    except Exception as e:
        logger.exception("Unexpected error during data load")
        if run:
            run.alert(title="Unexpected Error", text=str(e))
        sys.exit(1)

    finally:
        if run:
            wandb.finish()
            logger.info("W&B run finished")


if __name__ == "__main__":
    # Hydra will supply the `cfg` at runtime — ignore the lint warning here
    main()  # pylint: disable=no-value-for-parameter
