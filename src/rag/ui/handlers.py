"""Callbacks de Gradio: streaming de respuestas, ingesta de uploads, citas.

Toda la lógica "imperativa" de la UI vive aquí para que `gradio_app.py`
quede limitado a layout y wiring.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..application.condense import format_history
from ..application.ingest import add_documents_to_index
from ..bootstrap import AppServices
from ..config import settings
from ..infrastructure.loaders import load_documents


logger = logging.getLogger(__name__)

ERROR_MESSAGE = (
    "⚠️ Ha ocurrido un error al generar la respuesta. "
    "Verifica que Ollama esté corriendo e inténtalo de nuevo."
)


def format_sources(docs) -> str:
    if not docs:
        return ""
    lines = ["\n\n---\n**Fuentes**"]
    seen: set[tuple[str, int | None]] = set()
    for d in docs:
        src = d.metadata.get("source", "desconocido")
        page = d.metadata.get("page")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        page_str = f" (p.{page + 1})" if page is not None else ""
        lines.append(f"- `{src}`{page_str}")
    return "\n".join(lines)


async def respond(
    services: AppServices,
    message: str,
    history: list[dict],
    top_k: int,
    enable_reranker: bool,
):
    """Async generator: streamea tokens y añade citas al final."""
    try:
        bundle = services.get_bundle(int(top_k), bool(enable_reranker))

        # Reescritura de la pregunta sólo si hay historial previo.
        if history:
            history_str = format_history(history)
            context_question = (
                await bundle.condense.ainvoke(
                    {"history": history_str, "question": message}
                )
            ).strip() or message
            logger.debug(
                "Pregunta reescrita: %r → %r", message, context_question
            )
        else:
            context_question = message

        docs = await bundle.retriever.ainvoke(context_question)
        logger.info(
            "Consulta (top_k=%d, reranker=%s): %d docs recuperados",
            int(top_k),
            bool(enable_reranker),
            len(docs),
        )

        partial_text = ""
        async for chunk in bundle.chain.astream(
            {"question": message, "context_question": context_question}
        ):
            partial_text += chunk
            yield partial_text

        yield partial_text + format_sources(docs)
    except Exception:
        logger.exception("Error generando respuesta para pregunta: %r", message)
        yield ERROR_MESSAGE


def ingest_uploaded_file(services: AppServices, file_obj) -> str:
    """Copia el PDF subido a data/sources y lo añade al índice."""
    if file_obj is None:
        return ""

    try:
        src_path = Path(file_obj.name)
        dest = settings.data_dir / src_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_path, dest)

        new_docs = load_documents(settings.data_dir)
        new_docs = [d for d in new_docs if d.metadata.get("source") == src_path.name]
        add_documents_to_index(new_docs)

        services.reload_vectorstore()

        logger.info(
            "Upload ingerido: %s (%d fragmentos)", src_path.name, len(new_docs)
        )
        return f"Añadido **{src_path.name}** al índice ({len(new_docs)} fragmentos)."
    except Exception:
        logger.exception("Error ingiriendo upload: %r", getattr(file_obj, "name", None))
        return "⚠️ No se pudo añadir el documento. Revisa los logs."
