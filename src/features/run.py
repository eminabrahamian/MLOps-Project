"""
MLflow‐compatible feature‐engineering step with Hydra config and W&B logging.

Loads the validated dataset artifact, applies each configured transformer
from FEATURE_TRANSFORMERS, saves the engineered CSV, and logs artifacts and summary
metrics back to W&B.
"""
import sys
import logging
from datetime import datetime
from pathlib import Path
import tempfile

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
import pandas as pd

# ── Ensure your src/ package is on PYTHONPATH when MLflow spins up a new env
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.features.features import FEATURE_TRANSFORMERS

# Load any .env keys (e.g. WANDB_API_KEY)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("features")


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
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
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
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
            csvs = list(art_dir.glob("*.csv"))
            if not csvs:
                logger.error("No CSV found in validated_data artifact")
                run.alert(title="Feature Eng Error", text="Missing validated_data CSV")
                sys.exit(1)
            df = pd.read_csv(csvs[0])
        if df.empty:
            logger.warning("Validated DataFrame is empty")

        # 3) Apply transformers
        applied = []
        params = {}
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
        df.to_csv(out_path, index=False)
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
