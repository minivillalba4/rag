"""CLI: (re)construye el índice FAISS a partir de ``data/sources``.

Uso:
    python -m scripts.build_index
"""

import logging

from src.rag.application.ingest import build_index
from src.rag.config import settings
from src.rag.logging_config import configure_logging


logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    logger.info("Indexando %s → %s", settings.data_dir, settings.index_dir)
    logger.info("Modelo de embeddings: %s", settings.embed_model)
    vs = build_index()
    logger.info("Índice creado con %d vectores", vs.index.ntotal)


if __name__ == "__main__":
    main()
