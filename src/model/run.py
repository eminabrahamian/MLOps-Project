"""
Hydra-driven, MLflow-invokable modeling step.

Fetches the preprocessed dataset from W&B, runs the train/valid/test workflow,
persists all artifacts (raw splits, processed splits, pipeline, model),
evaluates performance, and logs everything to W&B.
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from datetime import datetime

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

import hydra
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.model.model import run_model_pipeline

log = logging.getLogger(__name__)


def _setup_logging(level: str = "INFO") -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Execute the modeling pipeline.

    Loads the config, initializes Weights & Biases, fetches processed
    data, runs model training, and logs artifacts and metrics.
    """
    # 1) Load environment variables (if any)
    env_path = cfg.get("env_file", ".env")
    try:
        load_dotenv(env_path)
    except Exception:
        # no .env in this project; ignore
        pass

    # 2) Structured logging
    _setup_logging(cfg.logging.level)

    log.info("Configuration\n%s", OmegaConf.to_yaml(cfg))

    # 3) Initialize W&B
    run_name = f"model_{datetime.now():%Y%m%d_%H%M%S}"
    wandb_run = wandb.init(
        project=cfg.main.wandb.project,
        entity=cfg.main.wandb.entity,
        job_type="model",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    log.info("Started W&B run: %s", wandb_run.name)

    try:
        # 4) Fetch processed data artifact from W&B
        data_art = wandb_run.use_artifact("validated_data:latest")
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = data_art.download(root=tmpdir)

            data_file = os.path.join(data_dir, "processed_data.xlsx")
            if os.path.isfile(data_file):
                df = pd.read_excel(data_file)
            else:
                split_art = wandb_run.use_artifact("processed_data:latest")
                split_dir = split_art.download(root=tmpdir)
                train_df = pd.read_excel(
                    os.path.join(split_dir, "train_processed.xlsx")
                )
                valid_df = pd.read_excel(
                    os.path.join(split_dir, "valid_processed.xlsx")
                )
                test_df = pd.read_excel(os.path.join(split_dir, "test_processed.xlsx"))
                df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

        if df.empty:
            log.warning("No data found in processed artifact.")

        # 5) Run the full model pipeline
        #    This will split, fit pipeline & model, evaluate,
        #    and save artifacts
        run_model_pipeline(df, OmegaConf.to_container(cfg, resolve=True))

        # 6) Log artifacts directory if desired
        art_cfg = cfg.get("artifacts", {})
        for name, path in {
            "preprocessor": Path(PROJECT_ROOT)
            / Path(art_cfg.get("preprocessing_pipeline")),
            "model": Path(PROJECT_ROOT) / Path(art_cfg.get("model_path")),
        }.items():
            if path:
                artifact = wandb.Artifact(name, type=name)
                artifact.add_file(path)
                wandb_run.log_artifact(artifact, aliases=[run_name])

    except Exception as e:
        log.exception("Unexpected error during modeling step")
        raise e
    finally:
        wandb_run.finish()
        log.info("W&B run finished")


if __name__ == "__main__":
    # Hydra will supply the `cfg` at runtime — ignore the lint warning here
    main()  # pylint: disable=no-value-for-parameter
