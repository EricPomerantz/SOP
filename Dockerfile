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

# Default target: the runtime image.
FROM runtime
