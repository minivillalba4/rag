"""Casos de uso del RAG: ask (pregunta→respuesta), condense (rewrite), ingest.

Esta capa orquesta los ports del dominio. Usa LangChain Runnables como
framework de composición, pero no expone tipos de LangChain en su API
pública salvo donde la UI los consume directamente para streaming.
"""

from .ask import RagBundle, build_ask_service
from .condense import build_condense_chain, format_history
from .ingest import add_documents_to_index, build_index
from .retrieval import build_retriever

__all__ = [
    "RagBundle",
    "add_documents_to_index",
    "build_ask_service",
    "build_condense_chain",
    "build_index",
    "build_retriever",
    "format_history",
]
