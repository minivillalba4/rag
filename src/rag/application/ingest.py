"""Caso de uso 'ingest': decide qué cargar y cómo trocear; delega lo pesado.

La *política* (directorio de fuentes, tamaño de chunk, estrategia de split)
vive aquí. La *mecánica* (leer PDFs, llamar al splitter, embeddings, FAISS)
está en `infrastructure/`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ..config import settings
from ..infrastructure.chunking import split_documents
from ..infrastructure.loaders import load_documents
from ..infrastructure.faiss_store import build_embeddings


logger = logging.getLogger(__name__)


def build_index(data_dir: Path | None = None, index_dir: Path | None = None) -> FAISS:
    """Carga documentos, los trocea, embebe y persiste el índice FAISS."""
    data_dir = data_dir or settings.data_dir
    index_dir = index_dir or settings.index_dir

    docs = load_documents(data_dir)
    chunks = split_documents(docs)
    logger.info("Indexando %d documentos → %d chunks", len(docs), len(chunks))
    vectorstore = FAISS.from_documents(chunks, build_embeddings())

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    logger.info(
        "Índice guardado en %s (%d vectores)", index_dir, vectorstore.index.ntotal
    )
    return vectorstore


def add_documents_to_index(
    new_docs: list[Document],
    index_dir: Path | None = None,
) -> FAISS:
    """Añade documentos a un índice existente (o lo crea) y lo persiste."""
    from ..infrastructure.faiss_store import load_or_build_vectorstore

    index_dir = index_dir or settings.index_dir
    vectorstore = load_or_build_vectorstore()

    chunks = split_documents(new_docs)
    vectorstore.add_documents(chunks)
    vectorstore.save_local(str(index_dir))
    logger.info(
        "Añadidos %d chunks al índice existente (%d vectores en total)",
        len(chunks),
        vectorstore.index.ntotal,
    )
    return vectorstore
