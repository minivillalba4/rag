"""Adapter: OllamaEmbeddings → domain.Embedder."""

from __future__ import annotations

from typing import Sequence

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from ..config import settings
from ..domain.ports import Embedder


def build_embeddings() -> OllamaEmbeddings:
    """Fabrica un cliente Ollama de embeddings leyendo de settings."""
    return OllamaEmbeddings(
        model=settings.embed_model,
        base_url=settings.ollama_base_url,
    )


class OllamaEmbedder(Embedder):
    """Adapter que implementa `Embedder` envolviendo un `Embeddings` de LangChain."""

    def __init__(self, embeddings: Embeddings | None = None) -> None:
        self._embeddings = embeddings or build_embeddings()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)
