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

import hydra
import wandb
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

# ensure we can import our package
PROJECT_ROOT = Path(__file__).resolve().parents[2]    # .../MLOps
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.model.model import run_model_pipeline  # noqa: E402

log = logging.getLogger(__name__)

def _setup_logging(level: str = "INFO") -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
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
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    os.environ["WANDB_ENTITY"] = cfg.wandb.entity
    run_name = f"model_{datetime.now():%Y%m%d_%H%M%S}"
    wandb_run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        job_type="model",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    log.info("Started W&B run: %s", wandb_run.name)

    try:
        # 4) Fetch preprocessed data artifact from W&B
        artifact = wandb_run.use_artifact(cfg.data.artifact, type=cfg.data.type)
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = artifact.download(root=tmpdir)
            # locate a single CSV or Excel
            files = list(Path(data_dir).glob("*.csv")) or list(Path(data_dir).glob("*.xlsx"))
            if not files:
                raise FileNotFoundError(f"No CSV/XLSX found in artifact {cfg.data.artifact}")
            raw_path = files[0]
        log.info("Downloaded preprocessed data to %s", raw_path)

        # 5) Load into DataFrame
        if raw_path.suffix == ".csv":
            df = pd.read_csv(raw_path)
        else:
            df = pd.read_excel(raw_path, engine="openpyxl")
        log.info("Loaded data shape: %s", df.shape)

        # 6) Run the full model pipeline
        #    This will split, fit pipeline & model, evaluate, and save artifacts
        run_model_pipeline(df, OmegaConf.to_container(cfg, resolve=True))

        # 7) Log artifacts directory if desired
        art_cfg = cfg.get("artifacts", {})
        for name, path in {
            "preprocessor": art_cfg.get("preprocessing_pipeline"),
            "model": art_cfg.get("model_path"),
        }.items():
            if path:
                wandb_run.log_artifact(wandb.Artifact(name, type=name), aliases=[run_name]).add_file(path)

    except Exception as e:
        log.exception("Unexpected error during modeling step")
        raise e
    finally:
        wandb_run.finish()
        log.info("W&B run finished")


if __name__ == "__main__":
    # Hydra will supply the `cfg` at runtime — ignore the lint warning here
    main()  # pylint: disable=no-value-for-parameter
