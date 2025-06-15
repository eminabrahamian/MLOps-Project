"""
main.py

Orchestrator for the end-to-end MLflow pipeline.  
Defines a single Hydra entrypoint that launches each step 
(data_load, data_validation, model, evaluation, inference)
as a separate MLflow project subrun, and tracks the overall run in Weights & Biases.
"""

import os
import sys
import tempfile
import logging
from datetime import datetime
from pathlib import Path

import hydra
import mlflow
import wandb
from omegaconf import DictConfig, OmegaConf

# canonical list of pipeline stages (must match directory names under src/)
PIPELINE_STEPS = [
    "data_loader",
    "data_validator",
    "model",
    "evaluation",
    "inference",
]

# only certain stages allow extra Hydra overrides passed through MLflow
STEPS_WITH_OVERRIDES = {"model"}

# configure simple console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("orchestrator")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Hydra entrypoint for the orchestrator.

    Expects a `config.yaml` at the project root with keys:
      main:
        WANDB_PROJECT:   name of the W&B project
        WANDB_ENTITY:    W&B entity/user
        steps:           CSV list of steps to run, or "all"
        hydra_options:   optional overrides to pass to individual steps
    """
    # 1) push W&B credentials into env
    os.environ["WANDB_PROJECT"] = cfg.main.wandb.project
    os.environ["WANDB_ENTITY"] = cfg.main.wandb.entity
    # 2) start a high-level W&B run
    run_name = f"orchestrator_{datetime.now():%Y%m%d_%H%M%S}"
    wandb_run = wandb.init(
        project=cfg.main.wandb.project,
        entity=cfg.main.wandb.entity,
        job_type="orchestrator",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    logger.info("Started W&B orchestrator run: %s", wandb_run.name)

    # 3) determine which steps to execute
    raw_steps = cfg.main.steps.strip()
    if raw_steps.lower() == "all":
        active = PIPELINE_STEPS
    else:
        active = [s.strip() for s in raw_steps.split(",") if s.strip()]

    # 4) grab any Hydra overrides to pass along
    override = getattr(cfg.main, "hydra_options", "")

    # 5) run each step in its own MLflow sub-run
    base_dir = hydra.utils.get_original_cwd()
    with tempfile.TemporaryDirectory():
        for step in active:
            logger.info("→ Running step: %s", step)
            step_path = Path(base_dir) / "src" / step
            if not step_path.exists():
                logger.error("Step directory not found: %s", step_path)
                sys.exit(1)

            # build MLflow parameters
            params = {}
            if override and step in STEPS_WITH_OVERRIDES:
                params["hydra_options"] = override

            try:
                mlflow.run(
                    uri=str(step_path),
                    entry_point="main",
                    parameters=params,
                    experiment_name=cfg.main.wandb.project,
                )
                logger.info("Completed step: %s", step)
            except Exception as e:
                logger.exception("Step %s failed", step)
                wandb_run.alert(
                    title="Orchestrator Failure",
                    text=f"Step `{step}` crashed: {e}"
                )
                sys.exit(1)

    # 6) wrap up
    wandb_run.finish()
    logger.info("Orchestrator run complete")


if __name__ == "__main__":
    # Hydra will supply the `cfg` at runtime — ignore the lint warning here
    main()  # pylint: disable=no-value-for-parameter
