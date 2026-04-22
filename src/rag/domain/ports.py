"""Contratos (ports) que las capas externas implementan.

La capa `application` depende SOLO de estas interfaces; los adapters
concretos viven en `infrastructure` y las instancia el composition root.
"""

from __future__ import annotations

from typing import AsyncIterator, Iterator, Protocol, Sequence, runtime_checkable

from .entities import ChatMessage, DocumentChunk


@runtime_checkable
class Embedder(Protocol):
    """Convierte texto a vectores densos."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Almacenamiento de chunks con búsqueda por similitud."""

    def similarity_search(self, query: str, k: int) -> list[DocumentChunk]: ...
    def add(self, chunks: Sequence[DocumentChunk]) -> None: ...
    def persist(self) -> None: ...


@runtime_checkable
class LLM(Protocol):
    """Modelo de chat. Streaming obligatorio para la UI."""

    def invoke(self, messages: Sequence[ChatMessage]) -> str: ...
    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]: ...
    async def ainvoke(self, messages: Sequence[ChatMessage]) -> str: ...
    def astream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]: ...


@runtime_checkable
class Reranker(Protocol):
    """Reordena candidatos por relevancia a la query (opcional)."""

    def rerank(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
        top_n: int,
    ) -> list[DocumentChunk]: ...
