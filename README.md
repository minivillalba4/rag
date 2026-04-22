---
title: RAG CV
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.3.0
app_file: app.py
pinned: false
---

# RAG sobre CV — asistente conversacional

Sistema RAG con arquitectura hexagonal que responde preguntas sobre el CV
del candidato citando las fuentes. Soporta dos modos de ejecución:

- **Local / on-prem** con Ollama (modelo y embeddings locales).
- **Desplegado en cloud gratis** con Groq (LLM) + HuggingFace embeddings
  locales en CPU — cabe en HuggingFace Spaces Free o Render Free sin
  tarjeta de crédito.

El *provider* se elige por variable de entorno sin tocar código:

```
LLM_PROVIDER=ollama | groq
EMBED_PROVIDER=ollama | huggingface
```

## Arquitectura

```
src/rag/
├── domain/           entidades + puertos (LLM, Embedder, VectorStore, Reranker)
├── application/      casos de uso (ingest, ask, retrieval, condense, prompts)
├── infrastructure/   adapters: Ollama, Groq, HuggingFace, FAISS, loaders, BGE
├── ui/               callbacks y layout Gradio
├── bootstrap.py      composition root + dispatch de providers
└── main.py           entrypoint local
app.py                entrypoint para HuggingFace Spaces y Render
```

## Arranque local con Docker (Ollama)

```bash
docker compose up --build
# http://localhost:7860
```

Primer arranque ≈ 10 min (descarga `qwen2.5:7b-instruct` + `nomic-embed-text`);
siguientes < 30 s. Si la descarga se corta, re-ejecuta `docker compose up`: la
descarga reanuda.

## Arranque local sin Docker

```bash
pip install -r requirements.txt

# Opción 1 — Ollama (requiere Ollama corriendo)
export LLM_PROVIDER=ollama
export EMBED_PROVIDER=ollama
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text

# Opción 2 — Groq + HF local (solo internet)
export LLM_PROVIDER=groq
export EMBED_PROVIDER=huggingface
export GROQ_API_KEY=gsk_...   # https://console.groq.com/keys

python -m scripts.generate_cv_pdf   # → data/sources/cv_candidato.pdf
python -m scripts.build_index        # → data/index/
python app.py                        # Gradio en :7860
```

## Despliegue en HuggingFace Spaces (gratis)

1. Crea un Space tipo **Gradio** en <https://huggingface.co/new-space>.
2. Sube el repo completo (o conéctalo a GitHub).
3. En *Settings → Variables and secrets* añade:
   - `LLM_PROVIDER = groq`
   - `EMBED_PROVIDER = huggingface`
   - `GROQ_API_KEY` (secret) = tu clave de <https://console.groq.com/keys>
4. El Space detecta `app.py` gracias al frontmatter YAML al inicio de este README.

Ventajas: 2 vCPU + 16 GB RAM gratis. Duerme tras 48 h sin tráfico y
despierta con el primer acceso.

## Despliegue en Render Free (gratis)

El repo incluye [render.yaml](render.yaml). En el dashboard de Render:

1. *New → Blueprint* → apunta al repo.
2. En *Environment* añade el secreto `GROQ_API_KEY`.
3. Deploy.

El plan Free tiene 512 MB de RAM (por eso el blueprint fija el embedder a
`all-MiniLM-L6-v2`, ~22 MB) y duerme tras 15 min sin tráfico.

## Tests

```bash
pytest -v
```

## Evaluación con RAGAS

```bash
pip install -r requirements-optional.txt
python -m scripts.run_eval --sample 3
```

## Configuración

Copia [.env.example](.env.example) a `.env` y ajusta. Ver
[src/rag/config.py](src/rag/config.py) para la lista completa.

## GPU (opcional, docker-compose)

Para Ollama con NVIDIA, añade al servicio `ollama`:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```
