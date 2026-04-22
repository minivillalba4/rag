"""Núcleo del dominio: entidades y contratos.

Sin dependencias de langchain/ollama/faiss. Todo lo que viva aquí debe
poder ejecutarse en un entorno mínimo de Python.
"""

from .entities import AnswerWithSources, ChatMessage, DocumentChunk, SourceDocument
from .ports import Embedder, LLM, Reranker, VectorStore

__all__ = [
    "AnswerWithSources",
    "ChatMessage",
    "DocumentChunk",
    "Embedder",
    "LLM",
    "Reranker",
    "SourceDocument",
    "VectorStore",
]
