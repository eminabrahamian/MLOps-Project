FROM python:3.10-slim

WORKDIR /code
ENV PYTHONPATH=/code/src

# install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install dependencies first to leverage layer caching
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# copy application code and configuration files
COPY app ./app
COPY src ./src
COPY configs ./configs
COPY models ./models
COPY main.py ./
COPY MLproject ./
COPY scripts ./scripts

# Create necessary directories
RUN mkdir -p data logs artifacts

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8000}/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
