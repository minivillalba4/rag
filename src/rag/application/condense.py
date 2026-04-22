"""Reescritura de preguntas con historial (history-aware query rewriting)."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from ..config import settings
from .prompts import condense_prompt_template


def format_history(history: list[dict[str, Any]], window: int | None = None) -> str:
    """Serializa los últimos N turnos a formato 'Usuario: ... / Asistente: ...'.

    Espera el formato de Gradio ``type="messages"``: lista de dicts con
    ``role`` ∈ {user, assistant} y ``content``.
    """
    window = window or settings.history_window
    if not history:
        return ""

    # Tomamos los últimos `window*2` mensajes (usuario + asistente por turno).
    recent = history[-window * 2 :]
    lines: list[str] = []
    for msg in recent:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        speaker = "Usuario" if role == "user" else "Asistente"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def build_condense_chain(llm: BaseChatModel) -> Runnable:
    """Devuelve un chain que reescribe ``{history, question} → pregunta autónoma``."""
    return condense_prompt_template | llm | StrOutputParser() | RunnableLambda(str.strip)
