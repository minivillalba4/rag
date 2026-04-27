"""
Orquestación del pipeline RAG.

Compone los módulos de cómputo (ingestion, chunking, embeddings, vectorstore,
retriever, context) con los módulos del LLM (guardrail, condense, generation)
en dos clases:

- `RagPipeline`            — single-turn Q&A. Sin historia, sin guardrail.
                             Útil para tests y benchmarks aislados.
- `ConversationalRagPipeline` — multi-turn con guardrail e historia. Es el
                             flujo principal de producción.

`build_rag_pipeline(config)` orquesta la fase de indexación (ingest →
chunks → embeddings → FAISS) y devuelve un `ConversationalRagPipeline`
listo para chatear.

Caché in-memory + `clear_pipeline_cache()` para `answer_question()`.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from src.chunking import split_documents
from src.condense import condense_question
from src.config import AppConfig, get_config
from src.context import build_context
from src.embeddings import build_embeddings
from src.footer import append_contact_footer
from src.generation import generate_answer, generate_answer_with_history
from src.guardrail import REFUSAL_MESSAGES, classify_intent
from src.ingestion import load_markdown_document
from src.retriever import normalize_sources, retrieve_documents
from src.vectorstore import build_vector_store


AnswerGenerator = Callable[[str, str], str]


@dataclass
class RagPipeline:
    """
    Single-turn Q&A. Sin historia ni guardrail.

    Mantiene contrato con `tests/test_pipeline.py`: `answer_generator` es
    inyectable para tests, y por defecto usa `generate_answer`.
    """

    vectorstore: Any
    answer_generator: AnswerGenerator = generate_answer

    def answer_question(self, question: str, k: int | None = None) -> dict:
        if k is None:
            k = get_config().retrieval_k
        docs = retrieve_documents(self.vectorstore, question, k=k)
        sources = normalize_sources(docs)
        context = build_context(sources)
        answer = self.answer_generator(context, question)
        return {"answer": answer, "sources": sources}


@dataclass
class ConversationalRagPipeline:
    """
    Multi-turn con guardrail de entrada y memoria propia.

    Su `chat(question)` orquesta:
        classify_intent → (si "allow") condense_question → retrieve →
        normalize → context → generate_answer_with_history.

    Si el guardrail bloquea, devuelve un mensaje fijo de refusal con footer
    (sin retrieval ni generación). Mutates `self.history` en ambos casos.
    """

    vectorstore: Any
    history: list = field(default_factory=list)

    def chat(self, question: str, k: int | None = None) -> dict:
        if k is None:
            k = get_config().retrieval_k

        intent = classify_intent(question)

        if intent["category"] != "allow":
            answer = append_contact_footer(REFUSAL_MESSAGES[intent["category"]])
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})
            return {
                "answer": answer,
                "sources": [],
                "standalone_question": question,
                "blocked": True,
                "intent": intent,
            }

        standalone = condense_question(self.history, question)
        docs = retrieve_documents(self.vectorstore, standalone, k=k)
        sources = normalize_sources(docs)
        context = build_context(sources)
        answer = generate_answer_with_history(context, question, self.history)

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": sources,
            "standalone_question": standalone,
            "blocked": False,
            "intent": intent,
        }

    def reset(self) -> None:
        self.history = []


# ---------------------------------------------------------------------------
# Construcción y caché
# ---------------------------------------------------------------------------


def build_rag_pipeline(config: AppConfig | None = None) -> ConversationalRagPipeline:
    """
    Orquesta la fase de indexación y devuelve un `ConversationalRagPipeline`
    listo para chatear. La carga de embeddings es costosa, así que conviene
    llamar a esto una sola vez al arrancar la aplicación.
    """
    cfg = config if config is not None else get_config()
    document = load_markdown_document(cfg.profile_path)
    chunks = split_documents([document])
    embeddings = build_embeddings()
    vectorstore = build_vector_store(chunks, embeddings)
    pipeline = ConversationalRagPipeline(vectorstore=vectorstore)
    _set_cached_pipeline(pipeline)
    return pipeline


_pipeline_cache: ConversationalRagPipeline | None = None


def _set_cached_pipeline(pipeline: ConversationalRagPipeline) -> None:
    global _pipeline_cache
    _pipeline_cache = pipeline


def get_cached_pipeline() -> ConversationalRagPipeline:
    """
    Devuelve el pipeline cacheado. Si nadie lo construyó todavía, lo
    construye con la configuración por defecto.
    """
    if _pipeline_cache is None:
        return build_rag_pipeline()
    return _pipeline_cache


def clear_pipeline_cache() -> None:
    """Limpia la caché interna. Lo usan los tests para forzar reconstrucción."""
    global _pipeline_cache
    _pipeline_cache = None


def answer_question(question: str) -> dict:
    """
    Helper procedural que usa el pipeline cacheado (single-turn Q&A puro).
    Mantiene compatibilidad con `tests/test_pipeline.py`.
    """
    cached = _pipeline_cache
    if cached is None:
        cached = build_rag_pipeline()
        _set_cached_pipeline(cached)
    rag = RagPipeline(vectorstore=cached.vectorstore, answer_generator=generate_answer)
    return rag.answer_question(question)
