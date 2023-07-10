FROM debian:bullseye AS base

# Base setup
RUN apt update
RUN apt upgrade -y
RUN apt install git build-essential python3 python3-pip -y
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt install git-lfs -y
RUN git lfs install

## Delete unnecessary cache files
RUN apt clean

# Build image
FROM base AS build

WORKDIR /

## Grab the llama.cpp repo
RUN git clone https://github.com/ggerganov/llama.cpp.git

## Grab the (massive) OpenLLaMa weights
RUN git clone https://huggingface.co/openlm-research/open_llama_7b

## Build llama.cpp and convert the open llama weights
WORKDIR /llama.cpp
RUN make
RUN python3 -m pip install -r requirements.txt
RUN python3 convert.py ../open_llama_7b

# Output image
FROM base as main

## Copy llama.cpp
WORKDIR /
COPY --from=build /llama.cpp llama.cpp

## Copy converted weights only from OpenLLaMa
RUN mkdir -p /llama.cpp/models/7B
COPY --from=build /open_llama_7b/ggml-model-f16.bin /llama.cpp/models/7B/ggml-model.bin

## Copy simple web server script
WORKDIR /llama.cpp
COPY server.py server.py

## Ports
EXPOSE 8765

WORKDIR /llama.cpp

ENTRYPOINT python3 server.py