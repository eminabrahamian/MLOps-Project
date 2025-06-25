"""
Unit tests for the MLflow pipeline orchestrator (main.py).

Covers:
- Full and partial step execution logic
- MLflow subproject triggering via mlflow.run
- W&B initialization and alerting
- Step directory existence checks
- Error handling and sys.exit behavior
- Hydra-driven config parsing and override propagation

Mocks all external dependencies (W&B, MLflow, Hydra, filesystem)
to ensure fast and isolated testing.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

import main  # root-level main.py


@pytest.fixture
def cfg_all_steps():
    return OmegaConf.create(
        {
            "main": {
                "wandb": {"project": "test_project", "entity": "test_entity"},
                "steps": "all",
                "hydra_options": "",
            }
        }
    )


@pytest.fixture
def cfg_some_steps():
    return OmegaConf.create(
        {
            "main": {
                "wandb": {"project": "test_project", "entity": "test_entity"},
                "steps": "model,inference",
                "hydra_options": "model.param=value",
            }
        }
    )


@patch("main.wandb.init")
@patch("main.mlflow.run")
@patch("main.hydra.utils.get_original_cwd", return_value="/project")
@patch("main.Path.exists", return_value=True)
def test_run_all_steps(mock_exists, mock_cwd, mock_mlflow, mock_wandb, cfg_all_steps):
    run_mock = MagicMock()
    mock_wandb.return_value = run_mock

    main.main(cfg_all_steps)

    assert mock_mlflow.call_count == len(main.PIPELINE_STEPS)
    for call in mock_mlflow.call_args_list:
        assert call.kwargs["entry_point"] == "main"
    run_mock.finish.assert_called_once()


@patch("main.wandb.init")
@patch("main.mlflow.run")
@patch("main.hydra.utils.get_original_cwd", return_value="/project")
@patch("main.Path.exists", return_value=True)
def test_run_selected_steps_with_override(
    mock_exists, mock_cwd, mock_mlflow, mock_wandb, cfg_some_steps
):
    run_mock = MagicMock()
    mock_wandb.return_value = run_mock

    main.main(cfg_some_steps)

    assert mock_mlflow.call_count == 2
    assert mock_mlflow.call_args_list[0].kwargs["parameters"] == {
        "hydra_options": "model.param=value"
    }
    run_mock.finish.assert_called_once()


@patch("main.wandb.init")
@patch("main.hydra.utils.get_original_cwd", return_value="/project")
@patch("main.Path.exists", return_value=True)
@patch("main.mlflow.run", side_effect=RuntimeError("MLflow failed"))
def test_step_failure_triggers_exit_and_alert(
    mock_mlflow, mock_exists, mock_cwd, mock_wandb
):
    cfg = OmegaConf.create(
        {
            "main": {
                "wandb": {"project": "test_project", "entity": "test_entity"},
                "steps": "model",
                "hydra_options": "",
            }
        }
    )

    wandb_mock = MagicMock()
    mock_wandb.return_value = wandb_mock

    with patch.object(sys, "exit", side_effect=SystemExit):
        with pytest.raises(SystemExit):
            main.main(cfg)

    wandb_mock.alert.assert_called_once()
    wandb_mock.finish.assert_not_called()
