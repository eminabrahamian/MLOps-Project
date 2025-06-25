"""
Orchestrates end-to-end model evaluation using Hydra configs.

Loads trained model and processed datasets, computes metrics for each split,
and saves JSON reports and scalars to Weights & Biases.
Includes structured logging, error handling, and configurable splits.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.evaluation.evaluation import generate_split_report


@hydra.main(
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """
    MLflow-/Hydra-ready wrapper for our evaluation module.

    Picks up parameters from cfg, runs generate_split_report for each split,
    and logs everything to W&B.
    """
    # 2) Load env vars if an .env file is specified (no error if missing)
    env_path = cfg.get("env_file", ".env")
    if Path(env_path).is_file():
        load_dotenv(env_path)

    # 3) Configure structured logging
    log_level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=cfg.logging.format,
        datefmt=cfg.logging.datefmt if "datefmt" in cfg.logging else None,
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation step")

    # 4) Initialize W&B
    run_name = f"evaluation_{datetime.now():%Y%m%d_%H%M%S}"
    try:
        wandb_run = wandb.init(
            project=cfg.main.wandb.project,
            entity=cfg.main.wandb.entity,
            job_type="evaluation",
            name=run_name,
        )
        # log full config to W&B
        wandb.config.update(
            OmegaConf.to_container(cfg, resolve=True), allow_val_change=True
        )
    except Exception as e:
        logger.error("Failed to start W&B run: %s", e)
        raise

    # 5) For each split, generate & save metrics
    try:
        splits = cfg.evaluation.splits if "splits" in cfg.evaluation else ["validation"]
        metrics_dir = Path(PROJECT_ROOT) / Path(cfg.artifacts.metrics_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        for split in splits:
            logger.info("Evaluating split '%s'", split)
            report = generate_split_report(
                config=OmegaConf.to_container(cfg, resolve=True),
                split=split,
                processed_dir=cfg.artifacts.processed_dir,
                model_path=cfg.artifacts.model_path,
                save_path=str(metrics_dir),
            )

            # 6a) Log the JSON file as a W&B artifact
            json_path = metrics_dir / f"{split}_metrics.json"
            if json_path.is_file():
                wandb.save(str(json_path))
                logger.info("Saved split report to %s", json_path)

                artifact = wandb.Artifact(f"{split}_metrics.json", type=f'split_report_{split}')
                artifact.add_file(str(json_path), name=f"{split}_metrics.json")
                wandb_run.log_artifact(artifact, aliases=['latest'])
            else:
                logger.warning("Expected metrics file not found: %s", json_path)

            # 6b) Flatten & log numeric metrics to W&B
            flat = {}
            for name, val in report.items():
                if isinstance(val, (int, float)):
                    flat[f"{split}/{name}"] = val
            if flat:
                wandb.log(flat)

    except Exception as e:
        logger.exception("Unexpected error during evaluation: %s", e)
        raise

    finally:
        wandb_run.finish()
        logger.info("W&B run finished")


if __name__ == "__main__":
    # Hydra will supply the `cfg` at runtime — ignore the lint warning here
    main()  # pylint: disable=no-value-for-parameter
