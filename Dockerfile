# Python version can be changed, e.g.
# FROM python:3.8
# FROM ghcr.io/mamba-org/micromamba:1.5.1-focal-cuda-11.3.1
FROM docker.io/python:3.12.1-slim-bookworm

LABEL org.opencontainers.image.authors="FedMed Demo" \
      org.opencontainers.image.title="FedMed Flower Server" \
      org.opencontainers.image.description="ChRIS plugin that launches the Flower coordinator"

ARG SRCDIR=/usr/local/src/fedmed-pl-server
WORKDIR ${SRCDIR}

COPY requirements.txt .
RUN --mount=type=cache,sharing=private,target=/root/.cache/pip pip install -r requirements.txt

COPY . .
ARG extras_require=none
RUN pip install .[${extras_require}] \
    && cd / && rm -rf ${SRCDIR}
WORKDIR /

EXPOSE 9091 9092 9093

CMD ["fedmed-pl-superlink"]
