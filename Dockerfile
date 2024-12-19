FROM pytorch/pytorch:2.5.0-cuda11.8-cudnn9-runtime@sha256:d15e9803095e462e351f097fb1f5e7cdaa4f5e855d7ff6d6f36ec4c2aa2938ea

# Weirdly this stopped working
# RUN apt-get update \
#     &&  apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /rslearn

COPY ./requirements.txt /rslearn/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./extra_requirements.txt /rslearn/extra_requirements.txt
RUN pip install --no-cache-dir -r extra_requirements.txt
COPY ./test_requirements.txt /rslearn/test_requirements.txt
RUN pip install --no-cache-dir -r test_requirements.txt

COPY ./ /rslearn
RUN pip install --no-cache-dir  rslearn[extra]
