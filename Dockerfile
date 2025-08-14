FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime@sha256:7db0e1bf4b1ac274ea09cf6358ab516f8a5c7d3d0e02311bed445f7e236a5d80

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /rslearn

COPY pyproject.toml /rslearn/pyproject.toml
COPY uv.lock /rslearn/uv.lock
RUN uv sync --extra extra --extra dev

ENV PATH="/rslearn/.venv/bin:$PATH"
COPY ./ /rslearn
