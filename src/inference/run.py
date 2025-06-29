"""
Hydra-driven, MLflow-invokable inference step.

Loads a trained model and preprocessing pipeline from W&B artifacts,
runs batch inference on input data, logs predictions and summaries back to W&B.
"""

import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.inference.inference import run_inference  # your inference entrypoint

# ── Helpers ────────────────────────────────────────────────────────────


def _df_hash(df: pd.DataFrame) -> str:
    """Compute a stable hash of a DataFrame's contents + index."""
    raw = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(raw).hexdigest()


# ── Entry point ────────────────────────────────────────────────────────


@hydra.main(
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Drive batch inference via MLflow + W&B."""
    # bring .env keys into env
    load_dotenv()

    # serialize Hydra cfg for reproducibility
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # prepare WandB run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"inference_{ts}"
    run = wandb.init(
        project=cfg.main.wandb.project,
        entity=cfg.main.wandb.entity,
        job_type="inference",
        name=run_name,
        config=cfg_dict,
        tags=["inference"],
    )
    logger = logging.getLogger("inference")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("Started W&B run: %s", run_name)

    try:
        # resolve input/output from config
        input_path = (PROJECT_ROOT / cfg.inference.input_file).resolve()
        output_path = (PROJECT_ROOT / cfg.inference.output_file).resolve()

        # try to pull preprocessed data artifact for lineage
        try:
            art = run.use_artifact("new_inference_data:latest")
            with tempfile.TemporaryDirectory() as tmpdir:
                raw_inference_data_path = art.download(root=tmpdir)
                df = pd.read_excel(
                    os.path.join(raw_inference_data_path, "new_inference_data.xlsx")
                )
                if df.empty:
                    logger.warning("Loaded DataFrame is empty; skipping validation")
                if df.duplicated().sum() > 0:
                    logger.warning(
                        "DataFrame contains duplicates;" " consider deduplication"
                    )
                logger.info("Downloaded inference data: %s", raw_inference_data_path)

        except Exception:
            logger.warning(
                "Could not fetch artifact '%s'; using %s",
                "new_inference_data:latest",
                input_path,
            )

        # log input hash and schema
        if input_path.is_file():
            df_in = (
                pd.read_csv(input_path)
                if input_path.suffix == ".csv"
                else pd.read_excel(input_path)
            )
            wandb.summary["input_data_hash"] = _df_hash(df_in)
            schema = {col: str(dt) for col, dt in df_in.dtypes.items()}
            wandb.summary["input_schema"] = schema
            # record schema file
            schema_path = (
                PROJECT_ROOT / "artifacts" / f"infer_input_schema_{run.id[:8]}.json"
            )
            schema_path.parent.mkdir(parents=True, exist_ok=True)
            schema_path.write_text(json.dumps(schema, indent=2))
            art_schema = wandb.Artifact("inference_input_schema", type="schema")
            art_schema.add_file(str(schema_path))
            run.log_artifact(art_schema, aliases=["latest"])
        else:
            logger.warning("Input file not found: %s", input_path)

        # save Hydra config snapshot
        cfg_file = PROJECT_ROOT / "artifacts" / f"infer_cfg_{run.id[:8]}.yaml"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_file, "w") as f:
            OmegaConf.save(config=cfg, f=f.name)
        wandb.save(str(cfg_file))

        # run actual inference
        t0 = time.time()
        run_inference(
            input_path=str(input_path),
            config_path=str(cfg_file),
            output_path=str(output_path),
            return_proba=cfg.inference.get("return_proba", False),
        )
        wandb.summary["inference_duration_s"] = time.time() - t0

        # log predictions artifact + table + summaries
        if Path(output_path).is_file():
            df_out = pd.read_excel(output_path)
            wandb.log({"predictions_table": wandb.Table(dataframe=df_out)})
            wandb.summary["n_predictions"] = len(df_out)
            wandb.summary["prediction_columns"] = list(df_out.columns)

            if "prediction_proba" in df_out.columns:
                probs = df_out["prediction_proba"]
                wandb.summary["proba_mean"] = float(probs.mean())
                wandb.summary["proba_min"] = float(probs.min())
                wandb.summary["proba_max"] = float(probs.max())

            # persist as artifact
            pred_art = wandb.Artifact("predictions", type="predictions")
            pred_art.add_file(str(output_path))
            run.log_artifact(pred_art, aliases=["latest"])
            logger.info("Logged predictions to W&B")

        logger.info(
            "Checking if input artifact logging is enabled: %s",
            cfg.inference.get("log_artifacts", True),
        )

        if cfg.inference.get("log_artifacts", True):
            logger.info("Artifact logging is enabled")
            logger.info("Input path being logged: %s", input_path)
            logger.info("File exists: %s", input_path.exists())
            input_art = wandb.Artifact("new_inference_data", type="dataset")
            input_art.add_file(str(input_path), name=input_path.name)
            run.log_artifact(input_art, aliases=["latest"])

    except Exception as e:
        logger.exception("Inference step failed: %s", e)
        if run is not None:
            run.alert(title="Inference Error", text=str(e))
        sys.exit(1)

    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("W&B run finished")


if __name__ == "__main__":
    # Hydra will supply the `cfg` at runtime — ignore the lint warning here
    main()  # pylint: disable=no-value-for-parameter
