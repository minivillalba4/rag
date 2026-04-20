"""CLI: (re)construye el índice FAISS a partir de ``data/sources``.

Uso:
    python -m scripts.build_index
"""

from src.rag.config import settings
from src.rag.ingest import build_index


def main() -> None:
    print(f"Indexando {settings.data_dir} → {settings.index_dir}")
    print(f"Modelo de embeddings: {settings.embed_model}")
    vs = build_index()
    print(f"Índice creado con {vs.index.ntotal} vectores.")


if __name__ == "__main__":
    main()
