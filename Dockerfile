FROM python:3.9-slim-buster

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu && \
    rm -rf /root/.cache/pip/*

COPY train.py /workspace/