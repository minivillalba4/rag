"""Wrapper del splitter: mecánica de trocear texto.

La *política* (tamaño, solapamiento) vive en `settings` y se decide en
`application/ingest.py`. Aquí solo implementamos el cómo.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings


def split_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Trocea documentos LangChain con `RecursiveCharacterTextSplitter`.

    Los parámetros son opcionales; si no se pasan se leen de `settings`.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
