FROM docker.io/python:3.12.1-slim-bookworm

LABEL org.opencontainers.image.authors="FedMed Demo" \
      org.opencontainers.image.title="FedMed Flower Server" \
      org.opencontainers.image.description="ChRIS plugin that launches the Flower coordinator"

ARG SRCDIR=/usr/local/src/fedmed-pl-server
WORKDIR ${SRCDIR}

COPY requirements.txt .
RUN --mount=type=cache,sharing=private,target=/root/.cache/pip pip install -r requirements.txt

# NEW: install ssh client + autossh for reverse tunnels
RUN apt-get update && apt-get install -y --no-install-recommends \
      openssh-client autossh ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
      openssh-client autossh ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# IMPORTANT: ensure UID 1001 exists for Apptainer/CUBE runtime
RUN if ! getent passwd 1001 >/dev/null 2>&1; then \
      useradd -u 1001 -m appuser; \
    fi

COPY . .
ARG extras_require=none
RUN pip install .[${extras_require}] \
    && cd / && rm -rf ${SRCDIR}
WORKDIR /

EXPOSE 9091 9092 9093

CMD ["fedmed-pl-superlink"]
