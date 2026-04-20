from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


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
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            loaded = TextLoader(str(path), encoding="utf-8").load()

        rel = path.relative_to(data_dir)
        for d in loaded:
            d.metadata["source"] = str(rel)
        docs.extend(loaded)

    if not docs:
        raise ValueError(
            f"No se encontraron documentos soportados ({SUPPORTED_EXTENSIONS}) "
            f"en {data_dir}"
        )

    return docs
