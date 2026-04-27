"""
Ingesta de documentos del disco.

Carga el `profile.md` (o cualquier markdown) como `Document` de LangChain
con metadatos mínimos de origen. Único punto de I/O sobre los ficheros
fuente del proyecto.
"""

from pathlib import Path

from langchain_core.documents import Document


def load_markdown_document(path) -> Document:
    """
    Lee un markdown del disco y lo envuelve en un `Document` de LangChain.

    Lanza `FileNotFoundError` si la ruta no existe — la falta del documento
    fuente es un fallo de configuración, no algo que el pipeline deba
    manejar silenciosamente.
    """
    markdown_path = Path(path)
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown document not found: {markdown_path}")
    content = markdown_path.read_text(encoding="utf-8")
    return Document(
        page_content=content,
        metadata={
            "source": str(markdown_path),
            "title": markdown_path.stem,
            "type": "markdown",
        },
    )
