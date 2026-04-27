"""
Chunking de documentos.

Wrapper fino sobre `RecursiveCharacterTextSplitter`. Existe como módulo
independiente para que la estrategia de chunking sea un único punto de
cambio (cambiar tamaño, overlap o splitter es una sola línea).
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_config


def split_documents(documents, chunk_size=None, chunk_overlap=None):
    """
    Trocea una lista de `Document` en chunks. Si no se pasan tamaños,
    usa los defaults de `AppConfig`.
    """
    config = get_config()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size if chunk_size is not None else config.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else config.chunk_overlap,
    )
    return splitter.split_documents(documents)
