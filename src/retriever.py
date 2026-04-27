"""
Recuperación de chunks relevantes desde el vectorstore.

Dos responsabilidades:
- `retrieve_documents`: consulta similaridad coseno al vectorstore.
- `normalize_sources`: convierte los `Document` recuperados en dicts
  serializables que el resto del pipeline (contexto, UI) consume.
"""

from src.config import get_config


def retrieve_documents(vectorstore, query: str, k: int | None = None):
    """Top-k chunks más similares a la query. `k` por defecto desde config."""
    if k is None:
        k = get_config().retrieval_k
    return vectorstore.similarity_search(query, k=k)


def normalize_sources(docs):
    """
    Convierte una lista de `Document` en dicts con forma estable:
    {id, title, content, metadata}. El campo `metadata.rank` indica el
    orden de relevancia (1 = más relevante).
    """
    sources = []
    for index, doc in enumerate(docs):
        metadata = dict(doc.metadata or {})
        title = metadata.get("title") or metadata.get("source") or "profile"
        sources.append(
            {
                "id": str(index),
                "title": str(title),
                "content": doc.page_content,
                "metadata": {**metadata, "rank": index + 1},
            }
        )
    return sources
