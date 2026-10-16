# SOP Vision runtime image: `sopvision serve` / `sopvision eval` with an NVIDIA GPU.
# All images use the same locked dependencies.

FROM node:24-slim AS web-base
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./

FROM web-base AS web-tests
CMD ["npm", "test"]

FROM web-base AS web-dev
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

FROM web-base AS web
RUN npm run build

FROM python:3.12-slim-bookworm AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# OpenCV (pulled in by Ultralytics) needs these at runtime.
# ca-certificates needed for SSL verification
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates --fresh

WORKDIR /app
ENV YOLO_AUTOINSTALL=false \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=60 \
    PYTHONHTTPSVERIFY=1

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --system-certs --trusted-host download-r2.pytorch.org --trusted-host files.pythonhosted.org
COPY src/ src/
COPY spikes/ spikes/
RUN uv sync --frozen --no-dev --system-certs --trusted-host download-r2.pytorch.org --trusted-host files.pythonhosted.org
# Baked-in defaults; docker-compose.yml mounts ./sops over this so edits apply without a rebuild.
COPY sops/ sops/
COPY --from=web /web/dist web/dist

ENV PATH=/app/.venv/bin:$PATH
EXPOSE 8000
ENTRYPOINT ["sopvision"]
CMD ["serve", "--host", "0.0.0.0"]

# Unit tests in the same environment plus the dev group: `docker compose --profile test run --rm tests`
FROM runtime AS test
RUN uv sync --frozen --system-certs --trusted-host download-r2.pytorch.org --trusted-host files.pythonhosted.org
ENTRYPOINT ["pytest", "-p", "no:cacheprovider"]
CMD []

FROM runtime AS tools
RUN uv sync --frozen --no-dev --group training --system-certs --trusted-host download-r2.pytorch.org --trusted-host files.pythonhosted.org
ENTRYPOINT ["python"]
CMD ["--help"]

# The worker service (sopvision.jobs.worker): the app plus laya-train, for jobs queued from the dashboard. Laya gets
# its own environment so its pins can't move the app's; corporate root CAs as in docker/laya.Dockerfile (CORP_CA_DIR).
FROM runtime AS worker
COPY --from=certs . /usr/local/share/ca-certificates/
RUN update-ca-certificates
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    HF_HOME=/cache/huggingface \
    USE_TF=0
RUN uv venv /opt/laya --python python3.12 \
    && uv pip install --python /opt/laya/bin/python --system-certs --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 \
    && uv pip install --python /opt/laya/bin/python --system-certs "laya[serve]==0.3.28" \
    && ln -s /opt/laya/bin/laya-train /usr/local/bin/laya-train
ENTRYPOINT ["python", "-m", "sopvision.jobs.worker"]
CMD []

# Default target: the runtime image.
FROM runtime
