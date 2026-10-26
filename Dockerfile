# ---------------------------------------------------------------------------
# The prediction service's image (built and run by compose.yaml).
#
# A Dockerfile is a PACKING LIST: Docker runs the steps top to bottom and
# saves the result as an image. What may be copied in at all is limited by
# .dockerignore (src/, configs/ and requirements-serving.txt -- nothing else).
# ---------------------------------------------------------------------------

# 1. A small official Python, pinned exactly: building twice gives the same box.
FROM python:3.11.17-slim-trixie

# 2. Python and pip settings for containers: no .pyc files, log lines appear
#    immediately, no pip download cache in the image, no upgrade nagging.
#    MLflow's usage statistics are off here too.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_TELEMETRY=true

WORKDIR /app

# 3. Install the packages BEFORE copying our code. Docker reuses a step's saved
#    result (a "layer") when nothing it depends on has changed -- so editing a
#    line of Python rebuilds in seconds, instead of re-installing everything.
COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt

# 4. Our code and settings: src/ (the service and the feature code it reuses)
#    and configs/ (model name, MLflow address, the validation contract).
COPY configs/ configs/
COPY src/ src/

# 5. Run as an ordinary user, not root: if the service were ever broken into,
#    the attacker would not be the container's administrator.
RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000

# 6. Every 30 s Docker asks the service "are you alive?" (GET /health), so
#    `docker compose ps` shows "healthy" -- not just "running". The longer
#    start period covers fetching the champion from MLflow at startup.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

# 7. Start the service. --host 0.0.0.0: accept connections from outside the
#    container (compose then decides who may reach it -- see compose.yaml).
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
