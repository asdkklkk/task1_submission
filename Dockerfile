FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

WORKDIR /app

COPY requirements.txt .

RUN apt-get update \
	&& apt-get install -y python3-pip \
	&& pip install --no-cache-dir --break-system-packages -r requirements.txt 

COPY task1.2_int.py .
