"""
Hydra-driven, MLflow-invokable entrypoint for the data_validation step.

Loads raw data from a W&B artifact (or local path), runs schema validation,
and logs both the cleaned dataset and the validation report as W&B artifacts.
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.data_validator.data_validator import setup_logger, validate_data


def html_schema_report(report: dict) -> str:
    """
    Generate an HTML summary of the validation report.

    Args:
        report: dict with keys:
          - "result": overall pass/fail
          - "errors": list of error messages
          - "warnings": list of warning messages
          - "details": dict mapping column names to
                       dicts of individual check results

    Returns:
        HTML string for display in W&B or a browser.
    """
    lines = []
    # Title and overall result
    lines.append("<h2>Validation Summary</h2>")
    result = report.get("result", "unknown").upper()
    lines.append(f"<p><strong>Overall result:</strong> {result}</p>")

    # Counts
    errs = report.get("errors", [])
    warns = report.get("warnings", [])
    lines.append(
        f"<p><strong>Errors:</strong> {len(errs)}  |"
        "  <strong>Warnings:</strong> {len(warns)}</p>"
    )

    # List out errors and warnings
    if errs:
        lines.append("<h3>Errors</h3><ul>")
        for msg in errs:
            lines.append(f"<li>{msg}</li>")
        lines.append("</ul>")

    if warns:
        lines.append("<h3>Warnings</h3><ul>")
        for msg in warns:
            lines.append(f"<li>{msg}</li>")
        lines.append("</ul>")

    # Detailed per-column check results
    details = report.get("details", {})
    if details:
        # Collect all check names
        check_names = set()
        for checks in details.values():
            check_names.update(checks.keys())

        # Build table header
        headers = ["Column"] + sorted(check_names)
        lines.append("<h3>Check Details</h3>")
        lines.append("<table border='1'><tr>")
        for h in headers:
            lines.append(f"<th>{h}</th>")
        lines.append("</tr>")

        # One row per column
        for col, checks in details.items():
            lines.append("<tr>")
            # Column name cell
            lines.append(f"<td>{col}</td>")
            # Each check cell
            for name in sorted(check_names):
                val = checks.get(name, "")
                lines.append(f"<td>{val}</td>")
            lines.append("</tr>")
        lines.append("</table>")

    # Join and return
    return "\n".join(lines)


@hydra.main(
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entrypoint for data validation.

    Expects in config.yaml under `main:`:
      wandb.project, wandb.entity  # W&B project and entity names

    And under `data_validation:`:
      raw_artifact     # name of the W&B artifact containing raw CSV
      report_path      # relative path for JSON report
      action_on_error  # "raise" or "warn"
      enabled          # true/false
      schema:          # list of column schemas
        - name: ...
          dtype: ...
          required: ...
          min: ...
          max: ...
          allowed_values: [...]
    """
    # 1. Load any .env settings if present (silently skip if missing)
    env_file = PROJECT_ROOT / cfg.get("env_file", ".env")
    if env_file.is_file():
        load_dotenv(str(env_file))

    # 2. Configure logging
    logger = setup_logger(OmegaConf.to_container(cfg.logging, resolve=True))
    logger.info("Starting data_validator step")

    run = None
    try:
        # 3. Start a W&B run
        run = wandb.init(
            project=cfg.main.wandb.project,
            entity=cfg.main.wandb.entity,
            name=f"data_validator_{datetime.now():%Y%m%d_%H%M%S}",
            config=dict(cfg),
            job_type="data_validator",
            tags=["data_validator"],
        )
        logger.info("Initialized W&B run: %s/%s", cfg.main.wandb.project, run.name)

        # 4. Fetch raw data artifact from W&B
        raw_art = run.use_artifact("raw_data:latest")
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_data_path = raw_art.download(root=tmpdir)
            df = pd.read_excel(os.path.join(raw_data_path, "cancer.xlsx"))
        if df.empty:
            logger.warning("Loaded DataFrame is empty; skipping validation")
        if df.duplicated().sum() > 0:
            logger.warning("DataFrame contains duplicates;" " consider deduplication")
        logger.info("Downloaded raw data: %s", raw_data_path)

        # 5. Run validation
        # First, massage config so report_path is absolute under project root
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        rpt_rel = cfg_dict["data_validation"].get(
            "report_path", "logs/validation_report.json"
        )
        abs_report = PROJECT_ROOT / rpt_rel
        # ensure directory exists
        abs_report.parent.mkdir(parents=True, exist_ok=True)
        cfg_dict["data_validation"]["report_path"] = str(abs_report)

        # Now run the validation
        validate_data(df, config=cfg)

        # 6. Log cleaned dataset as W&B artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            cleaned_path = Path(tmpdir) / "validated_data.xlsx"
            df.to_excel(cleaned_path, index=False)
            art_clean = wandb.Artifact("validated_data", type="dataset")
            art_clean.add_file(str(cleaned_path), name="validated_data.xlsx")
            run.log_artifact(art_clean, aliases=["latest"])
            logger.info("Validated data artifact logged")

        # 7. Log validation report as W&B artifact and summary
        if abs_report.is_file():
            art_report = wandb.Artifact("validation_report", type="report")
            art_report.add_file(str(abs_report), name="validation_report.json")
            run.log_artifact(art_report, aliases=["latest"])
            logger.info("Validation report artifact logged")

            # Optional: push summary metrics
            import json

            report = json.loads(abs_report.read_text())
            wandb.summary["validation_status"] = report.get("status", "unknown")
            wandb.summary["num_errors"] = len(report.get("errors", []))
            wandb.summary["num_warnings"] = len(report.get("warnings", []))

            html = html_schema_report(report)
            wandb.log({"validation_report": wandb.Html(html)})

    except Exception as exc:
        logger.exception("Data validation failed")
        # Signal failure in W&B UI
        if wandb.run is not None:
            run.alert(title="Validation Error", text=str(exc))
        sys.exit(1)
    finally:
        if wandb.run is not None:
            run.finish()
            logger.info("W&B run closed")


if __name__ == "__main__":
    # Hydra will supply the `cfg` at runtime — ignore the lint warning here
    main()  # pylint: disable=no-value-for-parameter
