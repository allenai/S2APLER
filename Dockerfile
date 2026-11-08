FROM python:3.11-bookworm

# install base packages
RUN apt-get clean \
    && apt-get update --fix-missing \
    && apt-get install -y \
    git \
    curl \
    gcc \
    g++ \
    build-essential \
    wget \
    awscli

WORKDIR /work

# install python packages
COPY requirements.in .

# copy over the data file
COPY data/path_config.json ./data/

# add the code as the final step so that when we modify the code
# we don't bust the cached layers holding the dependencies and
# system packages.
COPY s2apler/ s2apler/
COPY scripts/ scripts/
COPY tests/ tests/
COPY .flake8 .flake8
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.in

# s2and pins setuptools 78-81, where `setup.py develop` is deprecated; install through
# pyproject.toml instead. Dependencies are already installed from requirements.in.
COPY setup.py README.md ./
RUN pip install --no-deps -e .

CMD [ "/bin/bash" ]
