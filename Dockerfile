FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc g++ build-essential ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .


RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
      torch==2.1.0+cpu faiss-cpu && \
    pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 SMARTQA_INPUT=/data

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY . .

ENTRYPOINT ["smartqa"]
CMD ["--help"]
