FROM python:3.11-bookworm

# install base packages
RUN apt-get clean \
    && apt-get update --fix-missing \
    && apt-get install -y \
    git \
    curl \
    build-essential \
    pkg-config \
    awscli

# Building the native extension needs Rust; installing the locked environment needs uv.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

WORKDIR /work

# copy over the data file
COPY data/path_config.json ./data/

# add the code as the final step so that when we modify the code
# we don't bust the cached layers holding the dependencies and
# system packages.
COPY pyproject.toml uv.lock README.md LICENSE VERSION ./
COPY s2apler_rust/ s2apler_rust/
COPY s2apler/ s2apler/
COPY scripts/ scripts/
COPY tests/ tests/

RUN uv sync --frozen --extra dev
ENV PATH="/work/.venv/bin:${PATH}"

CMD [ "/bin/bash" ]
