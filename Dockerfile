FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/eval/golden_dataset.jsonl ./tests/eval/golden_dataset.jsonl

RUN mkdir -p /opt/seed && \
    python -m scripts.generate_cv_pdf --output /opt/seed/cv_candidato.pdf

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

RUN useradd -m -u 1000 rag && \
    mkdir -p /app/data/sources /app/data/index && \
    chown -R rag:rag /app /opt/seed

USER rag

EXPOSE 7860
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "src.rag.main"]
