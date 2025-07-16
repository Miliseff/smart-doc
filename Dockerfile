FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ build-essential wget curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

FROM python:3.11-slim

LABEL org.opencontainers.image.title="smart-doc CLI" \
      org.opencontainers.image.source="https://github.com/Miliseff/smart-doc" \
      org.opencontainers.image.description="Turn any folder of docs into a searchable QA knowledge base."

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SMARTQA_INPUT=/data

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY . .

VOLUME ["/data"]

ENTRYPOINT ["smartqa"]
CMD ["--help"]
