"""
Construcción del vectorstore.

Wrapper fino sobre `FAISS.from_documents`. El vectorstore vive en memoria;
no se persiste a disco (la ingesta se rehace al arrancar `app.py`).
"""

from langchain_community.vectorstores import FAISS


def build_vector_store(chunks, embeddings) -> FAISS:
    """Indexa los chunks en un FAISS in-memory usando los embeddings dados."""
    return FAISS.from_documents(chunks, embeddings)
