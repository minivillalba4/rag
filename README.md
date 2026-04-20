# RAG personal — Asistente sobre mi CV

> Un asistente conversacional que responde preguntas sobre mi experiencia
> profesional usando **únicamente** los documentos de mi CV y proyectos.
> Stack 100 % open source, un solo comando para arrancarlo.

![Python](https://img.shields.io/badge/python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Ollama](https://img.shields.io/badge/Ollama-local-orange)
![Gradio](https://img.shields.io/badge/Gradio-5.x-yellow)

---

## Por qué este proyecto

La mayoría de proyectos de portfolio "RAG sobre un PDF" son intercambiables.
Aquí le di la vuelta: **el RAG responde sobre mí**. Un recruiter abre el
enlace, pregunta *"¿qué experiencia tiene con LLMs?"* y obtiene una
respuesta citada con el documento fuente. El propio proyecto es la demo y
el pitch a la vez.

## Arquitectura

```
┌──────────┐   pregunta    ┌───────────────┐   top_k docs   ┌──────────────┐
│  Gradio  │ ────────────▶ │   LangChain   │ ─────────────▶ │   FAISS       │
│   Chat   │               │   LCEL chain  │                │ (persisted)  │
└────┬─────┘               └───────┬───────┘                └──────┬───────┘
     │                             │                                │
     │ stream tokens               │ prompt + context               │ embeddings
     │                             ▼                                ▼
     │                     ┌───────────────┐                 ┌──────────────┐
     └──────tokens─────────│  ChatOllama   │◀──── modelo ────│   Ollama      │
                           │ (qwen2.5:7b)  │                 │ (nomic-embed) │
                           └───────────────┘                 └──────────────┘
```

- **Ingesta**: PDFs + Markdown → `RecursiveCharacterTextSplitter` (500/50)
  → `OllamaEmbeddings` → FAISS (persistido en `data/index/`).
- **Retrieval**: top-k similitud; el `k` es ajustable desde la UI.
- **Memoria conversacional**: un sub-chain condensa la pregunta actual con
  los últimos turnos (*"¿y en qué empresa?"* → *"¿en qué empresa trabajó
  con LLMs?"*) antes de recuperar y responder. Si la pregunta ya es
  autocontenida, se omite el condensador.
- **Generación**: `ChatOllama` con prompt guardrail en español (no alucina;
  si no tiene info, lo dice). Cada afirmación lleva una cita inline `[N]`.
- **UI**: Gradio `ChatInterface` con streaming real (tokens según se
  generan) y panel de fuentes indexado: cada `[N]` de la respuesta se
  mapea al fragmento concreto (archivo + página + snippet) en el panel
  inferior, de forma que el lector sabe exactamente de dónde procede
  cada dato.

## Qué demuestra técnicamente

- Diseño de una cadena **LCEL** limpia (retriever → prompt → LLM → parser),
  testeable con componentes falsos sin necesidad de Ollama corriendo.
- **Streaming asíncrono** de extremo a extremo (`astream` de LangChain →
  generador async de Gradio).
- Persistencia de índice FAISS en disco (no se re-embebe en cada arranque).
- Configuración tipada con **pydantic-settings** (env vars con defaults).
- Empaquetado reproducible con **docker-compose** (dos servicios: Ollama +
  app; el `init` container descarga los modelos automáticamente).
- Tests unitarios con `FakeListChatModel` + `DeterministicFakeEmbedding`.

## Cómo correrlo

### Opción A: Docker (recomendado)

```bash
cp .env.example .env
docker compose up --build
# abrir http://localhost:7860
```

El primer arranque descarga `qwen2.5:7b-instruct` (~4.7 GB) y
`nomic-embed-text` (~270 MB). Después, el arranque es casi inmediato.

### Opción B: local

```bash
# 1) Instala Ollama: https://ollama.com
ollama serve &
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text

# 2) Entorno Python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3) Coloca tu CV en data/sources/ y genera el índice
python -m scripts.build_index

# 4) Lanza la app
python -m src.rag.app
```

### Cambiar el modelo

Edita `.env` (o pasa como env var):

```bash
OLLAMA_MODEL=llama3.1:8b-instruct   # alternativa sólida
EMBED_MODEL=bge-m3                  # multilingüe, mejor en ES
```

## Estructura del repo

```
src/rag/
├── config.py        # Settings con pydantic-settings
├── loaders.py       # PDF + Markdown loaders
├── ingest.py        # chunking + embeddings + FAISS save
├── vectorstore.py   # load_or_build (singleton)
├── prompts.py       # SYSTEM_PROMPT en ES con guardrails
├── chain.py         # build_rag_chain → LCEL (retriever | prompt | llm)
└── app.py           # Gradio Blocks + streaming + upload PDF
scripts/build_index.py
tests/test_chain.py
docker-compose.yml
Dockerfile
```

## Tests

```bash
pytest tests/
```

Los tests **no requieren** Ollama: usan `FakeListChatModel` y
`DeterministicFakeEmbedding` para verificar que el chain se construye y
responde correctamente.

## Decisiones de diseño (y trade-offs)

- **FAISS sobre Chroma / Qdrant**: para este tamaño (decenas/cientos de
  chunks) FAISS persistido a disco es más simple, cero infra extra. En
  producción real con actualizaciones frecuentes → Qdrant/Weaviate.
- **Ollama sobre HF Transformers**: evita gestionar pesos, cuantizaciones
  y dependencias GPU en la app. El contenedor de Ollama se encarga.
- **LCEL sobre `RetrievalQA` legacy**: LCEL compone mejor, soporta
  streaming de forma natural y es más fácil de testear.
- **Gradio sobre Streamlit/FastAPI+React**: iteración rapidísima de UI y
  streaming listo de fábrica; suficiente para portfolio.
- **`nomic-embed-text` sobre `all-MiniLM`**: mantiene todo el stack en
  Ollama (un solo runtime). Para un mejor recall en ES puro, `bge-m3` es
  mejor opción y está documentada como alternativa.

## Próximos pasos (si sigo iterando)

- Evaluación con `ragas` (faithfulness, answer relevancy).
- Reranker (`bge-reranker-v2-m3`) entre retrieval y LLM.
- Deploy en Hugging Face Spaces con fallback a Groq cuando no hay Ollama.
- ~~Chat con memoria conversacional multi-turno (rewriting de la pregunta).~~ ✅ Hecho.

## Licencia

MIT.
