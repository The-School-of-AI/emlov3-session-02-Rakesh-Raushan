FROM python:3.9-slim

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY train.py /workspace/

ENTRYPOINT ["python3", "train.py"]