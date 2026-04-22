"""Entidades del dominio RAG.

Dataclasses inmutables; no saben nada de langchain/ollama/faiss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "system"]


@dataclass(frozen=True)
class SourceDocument:
    """Documento original antes de trocear (PDF, MD, TXT…)."""

    text: str
    source: str                      # nombre relativo del fichero
    page: int | None = None          # páginas para PDFs multipágina
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    """Fragmento indexable: el input real de embeddings y retrieval."""

    text: str
    source: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatMessage:
    """Turno de conversación compatible con el formato Gradio `type=messages`."""

    role: Role
    content: str


@dataclass(frozen=True)
class AnswerWithSources:
    """Respuesta final del RAG con las fuentes que la sustentan."""

    answer: str
    sources: tuple[DocumentChunk, ...] = ()
