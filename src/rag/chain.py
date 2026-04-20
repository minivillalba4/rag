from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama

from .config import settings
from .prompts import condense_prompt_template, prompt_template
from .vectorstore import load_or_build_vectorstore


@dataclass
class RagBundle:
    """Agrupa los runnables del flujo RAG.

    - `chain` genera la respuesta final a partir de una pregunta (ya autocontenida).
    - `retriever` devuelve los documentos para construir citas en la UI.
    - `condense_chain` reescribe la pregunta usando el historial. La orquestación
      (llamar o no al condensador según haya historial) vive fuera, para que la UI
      pueda reutilizar la pregunta condensada tanto en el retriever como en el chain.
    """

    chain: Runnable
    retriever: VectorStoreRetriever
    condense_chain: Runnable


SNIPPET_MAX_CHARS = 180


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "desconocido")
        page = d.metadata.get("page")
        header = f"[{i}] {src}" + (f" (p.{page + 1})" if page is not None else "")
        parts.append(f"{header}\n{d.page_content.strip()}")
    return "\n\n".join(parts)


def _snippet(text: str, limit: int = SNIPPET_MAX_CHARS) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "…"


def format_sources(docs: list[Document]) -> str:
    """Renderiza las fuentes con el mismo índice [N] que `_format_context`.

    Cada entrada incluye archivo, página (si existe) y un snippet del fragmento
    recuperado, de modo que el lector pueda localizar el texto concreto que
    respalda cada cita inline de la respuesta.
    """
    if not docs:
        return ""
    lines = ["\n\n---", "**Fuentes**"]
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "desconocido")
        page = d.metadata.get("page")
        page_str = f" (p.{page + 1})" if page is not None else ""
        lines.append(f"**[{i}]** `{src}`{page_str}")
        lines.append(f"> {_snippet(d.page_content)}")
    return "\n".join(lines)


def format_history(history: list[dict], max_turns: int | None = None) -> str:
    """Convierte el historial de Gradio a texto plano para el condensador.

    Gradio usa `type="messages"` → lista de dicts `{"role": "user"|"assistant", "content": str}`.
    Se conservan solo los últimos `max_turns` intercambios (user+assistant) para
    evitar prompts desbordados.
    """
    if not history:
        return ""
    limit = max_turns if max_turns is not None else settings.history_max_turns
    trimmed = history[-(limit * 2):] if limit > 0 else history
    lines = []
    for msg in trimmed:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"Usuario: {content}")
        elif role == "assistant":
            lines.append(f"Asistente: {content}")
    return "\n".join(lines)


def build_rag_chain(
    vectorstore: FAISS | None = None,
    llm: BaseChatModel | None = None,
    top_k: int | None = None,
) -> RagBundle:
    vectorstore = vectorstore or load_or_build_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k or settings.top_k}
    )

    llm = llm or ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )

    chain: Runnable = (
        {
            "context": retriever | RunnableLambda(_format_context),
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    condense_chain: Runnable = (
        condense_prompt_template
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda s: s.strip().strip('"').strip())
    )

    return RagBundle(chain=chain, retriever=retriever, condense_chain=condense_chain)
