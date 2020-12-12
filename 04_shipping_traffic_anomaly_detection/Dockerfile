# Creates Docker image containing all requirements.
FROM ubuntu:18.04

ADD requirements.txt /requirements.txt
RUN apt update && apt install -y \
    python3 \
    python3-pip
RUN pip3 install -r requirements.txt

ENV PYTHONPATH=/home/src:$PYTHONPATH
WORKDIR /home
