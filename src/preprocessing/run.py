"""
Hydra-driven, MLflow-invokable preprocessing step.

Fetches the cleaned dataset from W&B, builds & applies the
preprocessing pipeline, writes processed splits, and logs
them as W&B artifacts.
"""

import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
import pickle

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.preprocessing.preprocessing import (
    build_preprocessing_pipeline,
    get_output_feature_names,
)


@hydra.main(
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """
    Orchestrate the preprocessing step with Hydra and Weights & Biases.

    1. Loads the validated dataset artifact from W&B.
    2. Builds and runs the sklearn preprocessing pipeline.
    3. Splits into train/validation/test sets.
    4. Writes processed CSVs to disk.
    5. Logs all processed artifacts back to W&B.

    Includes structured logging, robust error handling, and configurable
    data paths via the Hydra config.
    """
    # 1) Load any .env (if present)
    env_file = PROJECT_ROOT / cfg.get("env_file", ".env")
    if env_file.is_file():
        load_dotenv(str(env_file))

    # 2) Logger
    log_level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=cfg.logging.format,
        datefmt=cfg.logging.datefmt if "datefmt" in cfg.logging else None,
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting preprocessing step")

    run = None
    try:
        # 3) Init W&B run
        run = wandb.init(
            project=cfg.main.wandb.project,
            entity=cfg.main.wandb.entity,
            job_type="preprocess",
            name=f"preprocess_{datetime.now():%Y%m%d_%H%M%S}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        logger.info("W&B run %s initialized", run.name)

        # 4) Fetch validated data artifact
        val_art = run.use_artifact("engineered_data:latest")
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = Path(val_art.download(root=tmpdir))
            # assume a single Excel file named as in your validator step
            file_name = Path(cfg.data_source.processed_path).name
            df = pd.read_excel(downloaded / file_name)
        if df.empty:
            logger.warning("Validated DataFrame is empty")

        logger.info("Loaded validated data from %s", downloaded)

        # 5) Build & apply preprocessing pipeline
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        pipeline = build_preprocessing_pipeline(cfg_dict)
        X = pipeline.fit_transform(df)
        cols = get_output_feature_names(
            pipeline, df.columns.tolist(), cfg_dict
        )
        df_proc = pd.DataFrame(X, columns=cols)
        df_proc[cfg.target] = df[cfg.target]

        # 6) Split and write processed data
        proc_dir = PROJECT_ROOT / cfg.artifacts.processed_dir
        proc_dir.mkdir(parents=True, exist_ok=True)

        # full dataset
        full_path = proc_dir / "cancer_processed.xlsx"
        df_proc.to_excel(full_path, index=False)

        # train/valid/test
        split_cfg = cfg.data_split
        test_frac = split_cfg.test_size
        valid_frac = split_cfg.valid_size
        rnd = split_cfg.random_state

        train_df, temp = train_test_split(
            df_proc,
            test_size=test_frac + valid_frac,
            random_state=rnd,
            stratify=df_proc[cfg.target],
        )
        valid_df, test_df = train_test_split(
            temp,
            test_size=valid_frac / (test_frac + valid_frac),
            random_state=rnd,
            stratify=temp[cfg.target],
        )

        train_df.to_excel(proc_dir / "train_processed.xlsx", index=False)
        valid_df.to_excel(proc_dir / "valid_processed.xlsx", index=False)
        test_df.to_excel(proc_dir / "test_processed.xlsx", index=False)

        # 7) Log processed artifacts
        if cfg.preprocessing.log_artifacts:
            art = wandb.Artifact("processed_data", type="dataset")
            for f in proc_dir.glob("*.xlsx"):
                art.add_file(str(f), name=f.name)
            run.log_artifact(art, aliases=["latest"])
            logger.info("Logged preprocessed data artifact")

        # 8) Log preprocessing pipeline
        pp_path = PROJECT_ROOT / cfg.artifacts.get("preprocessing_pipeline", "models/preprocessing_pipeline.pkl")
        pp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pp_path.open("wb") as f:
            pickle.dump(pipeline, f)
        logger.info("Saved preprocessing pipeline to %s", pp_path)
        
        if cfg.preprocessing.log_pipeline:
            artifact = wandb.Artifact(
                "preprocessing_pipeline", type="pipeline"
            )
            artifact.add_file(str(pp_path))
            run.log_artifact(artifact, aliases=["latest"])
            logger.info("Logged preprocessing pipeline artifact to WandB")

    except Exception as exc:
        logger.exception("Preprocessing failed")
        if run:
            run.alert(title="Preprocess Error", text=str(exc))
        sys.exit(1)

    finally:
        if run:
            run.finish()
            logger.info("W&B run closed")


if __name__ == "__main__":
    # Hydra will supply the `cfg` at runtime — ignore the lint warning here
    main()  # pylint: disable=no-value-for-parameter
