from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from .config import settings


def load_or_build_vectorstore(index_dir: Path | None = None) -> FAISS:
    """Carga el índice FAISS persistido o lo construye si no existe."""
    index_dir = index_dir or settings.index_dir
    embeddings = OllamaEmbeddings(
        model=settings.embed_model,
        base_url=settings.ollama_base_url,
    )

    if (index_dir / "index.faiss").exists():
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    from .ingest import build_index

    return build_index(index_dir=index_dir)
