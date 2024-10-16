FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime@sha256:58a28ab734f23561aa146fbaf777fb319a953ca1e188832863ed57d510c9f197

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
RUN pip install --no-cache-dir  .
