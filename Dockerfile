FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /rslearn

COPY pyproject.toml /rslearn/pyproject.toml
COPY uv.lock /rslearn/uv.lock
RUN uv sync --extra extra --extra dev --extra terratorch --no-install-project

ENV PATH="/rslearn/.venv/bin:$PATH"
COPY ./ /rslearn
RUN uv sync --extra extra --extra dev --extra terratorch --locked
