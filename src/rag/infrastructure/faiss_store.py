"""Adapter FAISS: persistencia y `domain.VectorStore` sobre LangChain FAISS."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ..config import settings
from ..domain.entities import DocumentChunk
from ..domain.ports import VectorStore


logger = logging.getLogger(__name__)


def build_embeddings() -> Embeddings:
    """Despacha entre embeddings de Ollama y HuggingFace locales según config."""
    provider = settings.embed_provider.lower()
    if provider == "huggingface":
        from .hf_embeddings import build_hf_embeddings

        logger.info("Embeddings: HuggingFace local (%s)", settings.hf_embed_model)
        return build_hf_embeddings()
    if provider == "ollama":
        from .ollama_embeddings import build_embeddings as _build_ollama_embeddings

        logger.info("Embeddings: Ollama (%s)", settings.embed_model)
        return _build_ollama_embeddings()
    raise ValueError(f"EMBED_PROVIDER desconocido: {provider!r}")


def load_or_build_vectorstore(
    index_dir: Path | None = None,
    embeddings: Embeddings | None = None,
) -> FAISS:
    """Carga el índice FAISS persistido o lo construye si no existe."""
    index_dir = index_dir or settings.index_dir
    embeddings = embeddings or build_embeddings()

    if (index_dir / "index.faiss").exists():
        logger.info("Cargando índice desde %s", index_dir)
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    logger.info("Índice no encontrado en %s, construyendo desde cero", index_dir)
    # Import perezoso: application.ingest importa de vuelta esta capa, así
    # que rompemos el ciclo con un import local.
    from ..application.ingest import build_index

    return build_index(index_dir=index_dir)


def _to_chunk(doc: Document) -> DocumentChunk:
    meta = dict(doc.metadata)
    source = meta.pop("source", "desconocido")
    page = meta.pop("page", None)
    return DocumentChunk(text=doc.page_content, source=source, page=page, metadata=meta)


class FaissVectorStore(VectorStore):
    """Adapter `VectorStore` sobre un FAISS de LangChain."""

    def __init__(self, store: FAISS, index_dir: Path | None = None) -> None:
        self._store = store
        self._index_dir = index_dir or settings.index_dir

    @property
    def raw(self) -> FAISS:
        """Exposed para la capa de application mientras use Runnables LangChain."""
        return self._store

    def similarity_search(self, query: str, k: int) -> list[DocumentChunk]:
        return [_to_chunk(d) for d in self._store.similarity_search(query, k=k)]

    def add(self, chunks: Sequence[DocumentChunk]) -> None:
        docs = [
            Document(
                page_content=c.text,
                metadata={"source": c.source, "page": c.page, **c.metadata},
            )
            for c in chunks
        ]
        self._store.add_documents(docs)

    def persist(self) -> None:
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self._index_dir))
