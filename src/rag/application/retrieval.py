"""Construcción del retriever con reranker cruzado opcional.

Mantenemos la rama sin reranker como default para no forzar la dependencia
pesada (``sentence-transformers`` → ``torch``). Cuando se activa, envolvemos
el retriever base con ``ContextualCompressionRetriever`` + ``CrossEncoderReranker``.
"""

from __future__ import annotations

import logging

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

from ..config import settings


logger = logging.getLogger(__name__)


def build_retriever(
    vectorstore: FAISS,
    top_k: int | None = None,
    *,
    enable_reranker: bool | None = None,
    fetch_k: int | None = None,
    reranker_model: str | None = None,
) -> BaseRetriever:
    """Devuelve el retriever a usar por la cadena.

    - Si ``enable_reranker`` es False: retriever FAISS plano con ``k=top_k``.
    - Si es True: recupera ``fetch_k`` candidatos y reranquea a ``top_k`` con
      ``BAAI/bge-reranker-v2-m3`` (u otro modelo configurado).
    """
    top_k = top_k or settings.top_k
    enable_reranker = (
        settings.enable_reranker if enable_reranker is None else enable_reranker
    )

    if not enable_reranker:
        logger.debug("Retriever plano (top_k=%d, reranker=off)", top_k)
        return vectorstore.as_retriever(search_kwargs={"k": top_k})

    fetch_k = fetch_k or settings.retriever_fetch_k
    model_name = reranker_model or settings.reranker_model

    # Import perezoso: la construcción del cross-encoder arrastra torch.
    from langchain_classic.retrievers import ContextualCompressionRetriever

    from ..infrastructure.reranker_bge import build_cross_encoder_compressor

    logger.info(
        "Retriever con reranker: modelo=%s fetch_k=%d top_k=%d",
        model_name,
        fetch_k,
        top_k,
    )

    base = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
    compressor = build_cross_encoder_compressor(top_n=top_k, model_name=model_name)
    return ContextualCompressionRetriever(
        base_retriever=base,
        base_compressor=compressor,
    )
