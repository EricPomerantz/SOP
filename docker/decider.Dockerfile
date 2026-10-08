# Local Decider service: Mapika/decider-4b behind TypeSafe's POST /v1/systemone (docker compose --profile decider).
# Weights are not in the image: docker/decider_serve.py downloads the pinned revision into the hf_cache volume once.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# gcc: Triton (flash-linear-attention) compiles a small launcher at runtime. libgl/glib: OpenCV for the vision smoke test.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Corporate root CAs (TLS-intercepting proxies): put *.crt in CORP_CA_DIR (default: docker/certs, empty).
# Verification stays on; the proxy's CA is simply trusted like any other.
COPY --from=certs . /usr/local/share/ca-certificates/
RUN update-ca-certificates

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=60 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    HF_HOME=/cache/huggingface \
    TRITON_CACHE_DIR=/cache/triton

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --only-group decider --system-certs

COPY docker/decider_serve.py docker/
ENV PATH=/app/.venv/bin:$PATH
EXPOSE 8001
CMD ["python", "docker/decider_serve.py"]
