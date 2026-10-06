# syntax=docker/dockerfile:1
# Kev (github.com/jaredpalmer/kev): an open decision model with TypeSafe's /v1/systemone API, served locally in place
# of the hosted Jev. Used by docker-compose.kev.yml. Pinned to a reviewed commit; weights download to /hf on first start.
# Python 3.13: Kev pins it in .python-version and its uv.lock carries 3.13 wheels.
FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

ADD https://github.com/jaredpalmer/kev.git#5e42a7a03f28134853dd3ff77461457e921e5ec1 /opt/kev
WORKDIR /opt/kev
# Linux torch wheels from PyPI bundle CUDA, so this image runs on the GPU as is.
RUN uv sync --frozen --no-dev --extra serve

ENV PATH=/opt/kev/.venv/bin:$PATH \
    HF_HOME=/hf
EXPOSE 8009
CMD ["python", "-m", "kev.serve", "--run", "jaredpalmer/kev-4b", "--host", "0.0.0.0", "--port", "8009"]
