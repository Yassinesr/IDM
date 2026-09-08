# Optional: only for the custom-image path on Alaya NeW.
#
# The platform manual recommends starting from an official image and building a
# venv inside the Workshop (scripts/alaya/00_bootstrap_workshop.sh) - it is much
# less painful than pushing a custom image. Use this file only if you need a
# reproducible image in the private registry.
#
#   docker build -t idm-vton:local-1.0 .
#   docker login registry.hd-01.alayanew.com:8443
#   docker tag idm-vton:local-1.0 \
#       registry.hd-01.alayanew.com:8443/alayanew-<your-uuid>/idm-vton:v1.0
#   docker push registry.hd-01.alayanew.com:8443/alayanew-<your-uuid>/idm-vton:v1.0
#
# CUDA 11.8 to match torch 2.0.1+cu118 from environment.yaml. Ubuntu 22.04
# ships python 3.10 as its system python, so no deadsnakes PPA is needed.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Domestic apt mirror - the default archive.ubuntu.com is slow from CN.
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g; \
            s|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' \
        /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        build-essential git wget curl vim unzip bzip2 ca-certificates \
        openssh-server \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace

# torch first and on its own layer: it is the slowest thing to rebuild, and
# basicsr's setup.py imports torch, so requirements.txt fails without it.
RUN python -m pip install --no-cache-dir \
        torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /workspace/
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /workspace

# Some Workshop platforms attach over SSH.
RUN mkdir -p /var/run/sshd \
    && echo 'root:root' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd
EXPOSE 22

# Model weights are NOT baked in - they live on the PVC via $HF_HOME. Keeping
# them out keeps the image small enough to push over a home connection.
ENV HF_ENDPOINT=https://hf-mirror.com \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860
EXPOSE 7860

CMD ["/bin/bash"]
