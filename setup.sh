#!/usr/bin/env bash
set -e

# Install python dependencies listed in environment.yml
pip install \
    aiohttp==3.9.5 \
    aiohttp-retry==2.9.1 \
    anyio==4.2.0 \
    awscli==2.27.27 \
    black==25.1.0 \
    boto3==1.34.69 \
    botocore==1.34.69 \
    dvc==3.60.1 \
    dvc-s3==3.2.0 \
    fastapi==0.111.0 \
    flake8==7.2.0 \
    httpx==0.28.1 \
    hydra-core==1.3.2 \
    mlflow-skinny==2.12.1 \
    numpy==1.26.4 \
    openpyxl==3.1.2 \
    pandas==2.2.2 \
    pre-commit==3.7.0 \
    pydantic==2.7.1 \
    pytest==8.1.1 \
    pytest-cov==4.1.0 \
    pytest-dotenv==0.5.2 \
    python-dotenv==1.0.1 \
    pyyaml==6.0.1 \
    requests==2.31.0 \
    scikit-learn==1.4.2 \
    scipy==1.13.1 \
    tqdm==4.66.4 \
    typer==0.12.3 \
    uvicorn==0.29.0 \
    wandb==0.17.0

# Add src to PYTHONPATH for this session
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

echo "Environment ready. PYTHONPATH set to include src/"
