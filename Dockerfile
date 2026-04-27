FROM python:3.11-slim

LABEL org.opencontainers.image.title="RAG Portfolio Chatbot" \
      org.opencontainers.image.description="Chatbot RAG conversacional sobre el perfil profesional de Ismael Villalba"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False \
    PORT=7860

WORKDIR /app

# Dependencias de sistema:
# - build-essential, gcc: por si algún wheel no está disponible para la
#   arquitectura del contenedor y hay que compilar.
# - libgomp1: runtime de OpenMP que necesita faiss-cpu.
# build-essential se purga después para mantener la imagen pequeña.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Capa cacheable: solo se reconstruye si cambia requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Pre-descarga del modelo de embeddings para que el primer request no
# tenga que esperar al download (~90 MB). Sin esto, el cold start del
# Space añade 30-60s al primer turno.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copia del código (app.py, src/, data/profile/)
COPY . .

# Usuario no-root con UID 1000 (HF Spaces ejecuta el contenedor con este
# UID y monta volúmenes esperándolo). Propietario de /app para poder
# escribir en data/ (avatar del visitante) y en .cache/huggingface.
RUN useradd -m -u 1000 -s /bin/bash rag \
    && mkdir -p /app/.cache/huggingface \
    && chown -R rag:rag /app
USER rag

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/').status==200 else 1)"

CMD ["python", "app.py"]
