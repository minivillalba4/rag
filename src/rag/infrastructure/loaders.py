import logging
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}

logger = logging.getLogger(__name__)


def load_documents(data_dir: Path) -> list[Document]:
    """Carga todos los PDFs, markdown y textos de ``data_dir``.

    Devuelve una lista de ``Document`` con ``metadata["source"]`` apuntando
    al fichero original (nombre relativo), para poder mostrarlo como cita.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"No existe el directorio de documentos: {data_dir}"
        )

    docs: list[Document] = []
    per_ext: Counter[str] = Counter()
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
            if not loaded:
                logger.warning("PDF sin páginas legibles: %s", path)
        else:
            loaded = TextLoader(str(path), encoding="utf-8").load()

        rel = path.relative_to(data_dir)
        for d in loaded:
            d.metadata["source"] = str(rel)
        docs.extend(loaded)
        per_ext[path.suffix.lower()] += 1

    if not docs:
        raise ValueError(
            f"No se encontraron documentos soportados ({SUPPORTED_EXTENSIONS}) "
            f"en {data_dir}"
        )

    logger.info(
        "Documentos cargados desde %s: %d fragmentos brutos (ficheros por ext: %s)",
        data_dir,
        len(docs),
        dict(per_ext),
    )
    return docs
