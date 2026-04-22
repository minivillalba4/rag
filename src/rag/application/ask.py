"""Orquestador del caso de uso 'ask': pregunta → respuesta con fuentes.

Compone retrieval + prompt + LLM como un Runnable de LangChain. Mantiene
el historial-aware rewriter (condense) como componente separado para que
la UI pueda llamarlo antes del retrieval cuando hay historial.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from ..config import settings
from ..infrastructure.faiss_store import load_or_build_vectorstore
from ..infrastructure.ollama_llm import build_chat_llm
from .condense import build_condense_chain
from .prompts import prompt_template
from .retrieval import build_retriever


logger = logging.getLogger(__name__)


@dataclass
class RagBundle:
    """Agrupa el chain de respuesta, el retriever y el condenser.

    - ``chain`` genera la respuesta final; recibe un dict con ``question``
      (pregunta original, que aparece en el prompt) y ``context_question``
      (pregunta autónoma usada para el retrieval).
    - ``retriever`` se expone aparte porque la UI necesita los documentos
      fuente para mostrarlos como citas, separado del flujo de generación.
    - ``condense`` reescribe una pregunta con historial a forma autónoma.
      Recibe un dict con ``history`` (str) y ``question`` (str) y devuelve str.
    """

    chain: Runnable
    retriever: BaseRetriever
    condense: Runnable


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "desconocido")
        page = d.metadata.get("page")
        header = f"[{i}] {src}" + (f" (p.{page + 1})" if page is not None else "")
        parts.append(f"{header}\n{d.page_content.strip()}")
    return "\n\n".join(parts)


def _normalize(inp: Any) -> dict[str, str]:
    """Acepta str (pregunta sola) o dict con ``question``/``context_question``."""
    if isinstance(inp, str):
        return {"question": inp, "context_question": inp}
    question = inp.get("question", "")
    context_question = inp.get("context_question") or question
    return {"question": question, "context_question": context_question}


def build_ask_service(
    vectorstore: FAISS | None = None,
    llm: BaseChatModel | None = None,
    top_k: int | None = None,
    *,
    enable_reranker: bool | None = None,
    condense_llm: BaseChatModel | None = None,
    retriever: BaseRetriever | None = None,
) -> RagBundle:
    """Construye el servicio 'ask' con sus tres Runnables."""
    vectorstore = vectorstore or load_or_build_vectorstore()
    llm = llm or build_chat_llm()

    resolved_k = top_k or settings.top_k
    resolved_reranker = (
        settings.enable_reranker if enable_reranker is None else enable_reranker
    )
    retriever = retriever or build_retriever(
        vectorstore,
        top_k=resolved_k,
        enable_reranker=resolved_reranker,
    )

    logger.debug(
        "build_ask_service: top_k=%d reranker=%s", resolved_k, resolved_reranker
    )

    chain: Runnable = (
        RunnableLambda(_normalize)
        | {
            "context": RunnableLambda(lambda d: d["context_question"])
            | retriever
            | RunnableLambda(_format_context),
            "question": RunnableLambda(lambda d: d["question"]),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    condense = build_condense_chain(condense_llm or llm)

    return RagBundle(chain=chain, retriever=retriever, condense=condense)
