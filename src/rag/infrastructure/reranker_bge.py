"""Adapter: CrossEncoder (BGE) → domain.Reranker.

Imports perezosos: `sentence-transformers` arrastra torch y pesa >1 GB.
Sólo se cargan al instanciar el adapter, de modo que el resto de la app
arranca sin la dependencia.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ..config import settings
from ..domain.entities import DocumentChunk
from ..domain.ports import Reranker

if TYPE_CHECKING:
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker


def build_cross_encoder_compressor(
    top_n: int,
    model_name: str | None = None,
) -> "CrossEncoderReranker":
    """Devuelve un `CrossEncoderReranker` de LangChain listo para ContextualCompressionRetriever."""
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    return CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name=model_name or settings.reranker_model),
        top_n=top_n,
    )


class BgeCrossEncoderReranker(Reranker):
    """Adapter que implementa `Reranker` sobre HuggingFace CrossEncoder."""

    def __init__(self, model_name: str | None = None) -> None:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        self._encoder = HuggingFaceCrossEncoder(
            model_name=model_name or settings.reranker_model,
        )

    def rerank(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
        top_n: int,
    ) -> list[DocumentChunk]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self._encoder.score(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_n]]
