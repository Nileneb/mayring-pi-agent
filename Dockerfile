FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY mayring_pi_agent/ ./mayring_pi_agent/
# Installs mayring-pi-agent + its mayring-core git-subdirectory dependency.
RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1 \
    PI_PORT=8091

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8091/health')" || exit 1

CMD ["mayring-pi-agent"]
