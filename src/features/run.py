"""
MLflow‐compatible feature‐engineering step with Hydra config and W&B logging.

Loads the validated dataset artifact, applies each configured transformer
from FEATURE_TRANSFORMERS, saves the engineered XLSX, and logs artifacts and summary
metrics back to W&B.
"""
import sys
import logging
from datetime import datetime
from pathlib import Path
import tempfile

from dotenv import load_dotenv
import pandas as pd
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.features.features import FEATURE_TRANSFORMERS

import hydra
import wandb

# Load any .env keys (e.g. WANDB_API_KEY)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("features")


@hydra.main(config_path=str(PROJECT_ROOT / "configs"),\
            config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Entry point for feature engineering.

    Steps:
      1. Init W&B run
      2. Fetch validated data artifact
      3. Loop through cfg.features.engineered, apply each transformer
      4. Save engineered dataset and log it as a W&B artifact
      5. Log summary metrics (n_rows, n_cols, applied_features, feature_params)
    """
    # Convert to plain dict for W&B
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_name = f"features_{datetime.now():%Y%m%d_%H%M%S}"

    run = None
    try:
        # 1) Start W&B
        run = wandb.init(
            project=cfg.main.wandb.project,
            entity=cfg.main.wandb.entity,
            job_type="features",
            name=run_name,
            config=cfg_dict,
            tags=["features"],
        )
        logger.info("Started W&B run: %s", run_name)

        # 2) Download validated data artifact
        val_art = run.use_artifact("validated_data:latest")
        with tempfile.TemporaryDirectory() as tmp_dir:
            art_dir = Path(val_art.download(root=tmp_dir))
            excels = list(art_dir.glob("*.xlsx"))
            if not excels:
                logger.error("No Excel file (.xlsx) found in validated_data artifact")
                run.alert(title="Feature Eng Error", text="Missing validated_data Excel file")
                sys.exit(1)
            df = pd.read_excel(excels[0])
        if df.empty:
            logger.warning("Validated DataFrame is empty")

        # 3) Apply transformers (if enabled)
        applied = []
        params = {}
        if not cfg.features.get("enabled", True):
            logger.info("Feature engineering is disabled. Skipping all transformers.")
        else:
            for feat in cfg.features.get("engineered", []):
                builder = FEATURE_TRANSFORMERS.get(feat)
                if builder is None:
                    logger.debug("Skipping unregistered transformer: %s", feat)
                    continue
                transformer = builder(cfg_dict)
                df = transformer.transform(df)
                applied.append(feat)
                if hasattr(transformer, "get_params"):
                    params[feat] = transformer.get_params()
                logger.info("Applied transformer: %s", feat)

        # 4) Save engineered data
        out_path = PROJECT_ROOT / cfg.data_source.processed_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(out_path, index=False)
        logger.info("Saved engineered data to %s", out_path)

        # 4a) Log it as a W&B artifact
        if cfg.features.get("log_artifacts", True):
            art = wandb.Artifact("engineered_data", type="dataset")
            art.add_file(str(out_path))
            run.log_artifact(art, aliases=["latest"])
            logger.info("Logged engineered data artifact")

        # 5) Summary metrics
        wandb.summary.update({
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "applied_features": applied,
            "feature_params": params,
        })

    except Exception as e:
        logger.exception("Feature engineering failed: %s", e)
        if run:
            run.alert(title="Feature Eng Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run:
            wandb.finish()
            logger.info("Finished W&B run")


if __name__ == "__main__":
    # Hydra will supply the `cfg` at runtime — ignore the lint warning here
    main()  # pylint: disable=no-value-for-parameter
