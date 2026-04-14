FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt* pyproject.toml* ./
RUN pip install -U pip && \
    (pip install -r requirements.txt || true) && \
    pip install fastapi uvicorn[standard] pydantic redis httpx

COPY . .

ENV DMAPO_SERVE_HOST=0.0.0.0 \
    DMAPO_SERVE_PORT=8000 \
    DMAPO_CACHE_BACKEND=memory

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
