FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

ARG WANDB_API_KEY

ENV DEBIAN_FRONTEND=noninteractive \
    WANDB_API_KEY=${WANDB_API_KEY}

RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update -yqq && \
    apt-get install -yqq --no-install-recommends \
    git build-essential wget software-properties-common \
    python3 python3-pip python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -yqq && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -yqq --no-install-recommends python3.9 python3.9-dev python3.9-distutils && \
    python3.9 -m pip install --upgrade pip && \
    ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/local/pip3.9 /usr/bin/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements_pinned.txt /tmp
RUN pip install -r /tmp/requirements_pinned.txt && \
    rm /tmp/requirements_pinned.txt

WORKDIR /workspace
COPY . /workspace/
