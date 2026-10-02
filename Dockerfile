# SOP Vision runtime image: `sopvision serve` / `sopvision eval` with an NVIDIA GPU.
# Same uv.lock as the native run, so package versions match exactly.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# OpenCV (pulled in by Ultralytics) needs these at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ src/
RUN uv sync --frozen --no-dev
# Baked-in defaults; docker-compose.yml mounts ./sops over this so edits apply without a rebuild.
COPY sops/ sops/

ENV PATH=/app/.venv/bin:$PATH
EXPOSE 8000
ENTRYPOINT ["sopvision"]
CMD ["serve", "--host", "0.0.0.0"]
