FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime@sha256:7db0e1bf4b1ac274ea09cf6358ab516f8a5c7d3d0e02311bed445f7e236a5d80

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /rslearn

COPY pyproject.toml /rslearn/pyproject.toml
COPY uv.lock /rslearn/uv.lock
RUN uv sync --all-extras --no-install-project

ENV PATH="/rslearn/.venv/bin:$PATH"
COPY ./ /rslearn
RUN uv sync --all-extras --locked
