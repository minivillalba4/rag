"""Callbacks de Gradio: streaming de respuestas y citas.

Toda la lógica "imperativa" de la UI vive aquí para que `gradio_app.py`
quede limitado a layout y wiring.
"""

from __future__ import annotations

import logging

from ..application.condense import format_history
from ..bootstrap import AppServices
from ..config import settings


logger = logging.getLogger(__name__)

ERROR_MESSAGE = (
    "⚠️ Ha ocurrido un error al generar la respuesta. Inténtalo de nuevo "
    "en unos segundos; si persiste, puede ser un límite de cuota del "
    "proveedor."
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


async def respond(services: AppServices, message: str, history: list[dict]):
    """Async generator: streamea tokens y añade citas al final."""
    try:
        bundle = services.get_bundle(settings.top_k, settings.enable_reranker)

        if history:
            history_str = format_history(history)
            rewritten = await bundle.condense.ainvoke(
                {"history": history_str, "question": message}
            )
            context_question = (rewritten or "").strip() or message
            logger.debug("Pregunta reescrita: %r → %r", message, context_question)
        else:
            context_question = message

        docs = await bundle.retriever.ainvoke(context_question)
        logger.info(
            "Consulta (top_k=%d, reranker=%s): %d docs recuperados",
            settings.top_k,
            settings.enable_reranker,
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
