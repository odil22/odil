# hash:sha256:a502ef54ade3ed12bc199fc42b38892c380ae8c0308ae67e6e6fdc6fa88160ea
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libmpich-dev=3.3.2-2build1 \
        rsync=3.1.3-8ubuntu0.3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    matplotlib==3.5.2 \
    mpi4py==3.1.3 \
    numpy==1.22.3 \
    pyamg==4.2.3 \
    scipy==1.8.0 \
    tensorflow==2.8.0
