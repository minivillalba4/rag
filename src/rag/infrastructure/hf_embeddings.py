"""Adapter: HuggingFaceEmbeddings locales -> domain.Embedder.

Ejecuta el modelo en CPU dentro del propio proceso, sin llamadas
externas. `all-MiniLM-L6-v2` ocupa ~22 MB y cabe en los 512 MB
de Render Free. `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB)
da mejor calidad para consultas en español si hay RAM disponible.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from ..config import settings


def build_hf_embeddings() -> HuggingFaceEmbeddings:
    """Fabrica embeddings locales con el modelo configurado en settings."""
    return HuggingFaceEmbeddings(
        model_name=settings.hf_embed_model,
        encode_kwargs={"normalize_embeddings": True},
    )
