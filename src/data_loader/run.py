"""
src/data_loader/run.py

Hydra‐driven data‐loading step with robust try/except, structured logging,
and optional artifact tracking via Weights & Biases.

This script ties together:
- Configuration loading (`load_config`)
- Logger setup (`setup_logger`)
- Data ingestion (`load_data`)
from data_loader.py :contentReference[oaicite:0]{index=0}
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import hydra
import wandb
from omegaconf import DictConfig
from dotenv import load_dotenv

from data_loader import load_config, setup_logger, load_data, DataLoaderError


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Entry point: python run.py

    Expects in configs/config.yaml:
      data_source, logging, data_load (options), wandb (project/entity), env_file
    """
    # 1) Load environment variables
    env_path = cfg.get("env_file", ".env")
    load_dotenv(env_path)
    # 2) Set up structured logging
    logger = setup_logger(cfg.logging)
    logger.info("Loaded environment from %s", env_path)

    # 3) Initialize W&B run if configured
    run = None
    if cfg.get("wandb", {}).get("project"):
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"data_load_{datetime.now():%Y%m%d_%H%M%S}",
            config=cfg,
            job_type="data_load",
        )
        logger.info("Initialized W&B run: %s/%s", cfg.wandb.entity, cfg.wandb.project)

    try:
        # 4) Load data using your data_loader pipeline
        df = load_data()
        n_rows, n_cols = df.shape
        logger.info("Data loaded: %d rows, %d cols", n_rows, n_cols)

        # 5) Log basic metrics
        if run:
            wandb.log({"n_rows": n_rows, "n_cols": n_cols})

        # 6) Optionally log a sample of the data
        if cfg.data_load.get("log_sample", True) and run:
            sample_n = cfg.data_load.get("sample_n", 100)
            sample_tbl = wandb.Table(dataframe=df.head(sample_n))
            wandb.log({"sample_data": sample_tbl})
            logger.info("Logged sample of %d rows to W&B", sample_n)

        # 7) Optionally save raw data as an artifact
        if cfg.data_load.get("log_artifact", True) and run:
            artifact = wandb.Artifact("raw_data", type="dataset")
            raw_path = Path(cfg.data_source.raw_path)
            artifact.add_file(str(raw_path), name=raw_path.name)
            run.log_artifact(artifact)
            logger.info("Logged raw data artifact: %s", raw_path.name)

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
    main()
