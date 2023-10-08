FROM ubuntu:20.04

RUN apt update && \
    apt install -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install transformers

WORKDIR /app