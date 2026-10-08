# Laya (convaiinnovations/laya): 421M decision model behind TypeSafe's POST /v1/systemone (docker compose --profile laya).
# Evaluation only so far. Checkpoints download into the hf_cache volume on first start, pinned by LAYA_REVISION.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
# Corporate root CAs, as in docker/decider.Dockerfile (CORP_CA_DIR; default docker/certs, empty).
COPY --from=certs . /usr/local/share/ca-certificates/
RUN update-ca-certificates

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    HF_HOME=/cache/huggingface \
    USE_TF=0

# Same torch build as the rest of the project (CUDA 12.8), then Laya pinned.
RUN uv pip install --system --system-certs --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 \
    && uv pip install --system --system-certs "laya[serve]==0.3.28"

COPY docker/laya_serve.py /app/laya_serve.py
EXPOSE 8000
# Plain laya-serve unless LAYA_CHECKPOINT names a fine-tuned checkpoint (see docker/laya_serve.py).
CMD ["python", "/app/laya_serve.py"]
