FROM python:3.8-buster

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

RUN pip install -r requirements.in

COPY setup.py .
RUN python setup.py develop

CMD [ "/bin/bash" ]
