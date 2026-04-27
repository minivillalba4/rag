---
title: RAG Portfolio Chatbot
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Chatbot RAG conversacional sobre mi perfil profesional
---

# RAG Portfolio Chatbot

Chatbot RAG conversacional que responde, en primera persona, preguntas de un reclutador sobre mi perfil profesional. Recupera contexto desde un `profile.md` con FAISS + embeddings, mantiene historial entre turnos y filtra peticiones fuera de alcance con un guardrail LLM-as-judge.

## Stack técnico

- **UI**: Gradio 6 (`ChatInterface` con streaming).
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` vía LangChain.
- **Vectorstore**: FAISS in-memory.
- **LLM**: `meta-llama/Llama-3.3-70B-Instruct` vía Hugging Face Inference Providers (Cerebras como primario, ~2000 t/s; Groq como fallback, ~500-800 t/s).
- **Guardrail**: clasificador LLM-as-judge con política fail-open, categoriza cada pregunta en `allow / off_topic / tone_or_persona / prompt_injection`.

## Arquitectura

```
rag/
├── app.py                # Punto de entrada único — 6 líneas, lectura lineal.
├── src/
│   ├── config.py         # Muro de config: único módulo que lee .env.
│   ├── logging_setup.py  # Loggers `inference` y `guardrail`.
│   ├── prompts.py        # SYSTEM, CONDENSE, CLASSIFIER prompts.
│   ├── ingestion.py      # load_markdown_document
│   ├── chunking.py       # split_documents
│   ├── embeddings.py     # build_embeddings
│   ├── vectorstore.py    # build_vector_store (FAISS)
│   ├── retriever.py      # retrieve_documents, normalize_sources
│   ├── context.py        # build_context
│   ├── llm_client.py     # init_clients + chat_completion_with_fallback
│   ├── footer.py         # append_contact_footer
│   ├── generation.py     # generate / generate_with_history / stream_with_history
│   ├── condense.py       # condense_question
│   ├── guardrail.py      # classify_intent + REFUSAL_MESSAGES
│   ├── pipeline.py       # RagPipeline + ConversationalRagPipeline
│   └── ui/
│       ├── theme.py      # Paleta slate + azul, build_theme, CUSTOM_CSS
│       ├── avatars.py    # build_avatars
│       └── app.py        # gradio_chat, build_demo, launch_demo
└── tests/                # Suite pytest
```

Principios: SRP por módulo, KISS (sin ABCs ni factories), un único orquestador (`app.py`) y un muro de configuración (`src/config.py`).

## Variables de entorno

Configurar en el Space (Settings → Variables and secrets) o en un `.env` local:

| Variable | Obligatoria | Descripción |
|---|---|---|
| `HF_TOKEN` | Sí | Token de Hugging Face (Inference Providers). |
| `CONTACT_EMAIL` | No | Email que se inyecta en el footer de cada respuesta. |
| `CONTACT_LINKEDIN_URL` | No | URL de LinkedIn que se inyecta en el footer. |
| `PORT` | No | Puerto del servidor Gradio (default 7860). |
| `GRADIO_SERVER_NAME` | No | Bind address (default `0.0.0.0`). |

## Ejecución local

Con un entorno Python ≥3.11 y las dependencias instaladas:

```bash
pip install -r requirements.txt
python app.py
```

Abre `http://localhost:7860`.

## Ejecución con Docker

```bash
docker build -t rag-portfolio .
docker run --rm -p 7860:7860 --env-file .env rag-portfolio
```

## Tests

```bash
pytest tests/
```

## Licencia

MIT.
