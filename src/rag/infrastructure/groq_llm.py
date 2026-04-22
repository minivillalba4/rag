"""Adapter: ChatGroq -> domain.LLM.

Permite usar la API de Groq (free tier disponible) como reemplazo
hosteado de Ollama, ideal para desplegar sin GPU/RAM dedicadas.
"""

from __future__ import annotations

import os

from langchain_groq import ChatGroq

from ..config import settings


def build_groq_llm(temperature: float = 0.2) -> ChatGroq:
    """Fabrica un ChatGroq leyendo modelo y API key de settings/env."""
    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY no configurado. Obtén una clave gratis en "
            "https://console.groq.com/keys y expórtala como variable de entorno."
        )
    return ChatGroq(
        model=settings.groq_model,
        api_key=api_key,
        temperature=temperature,
    )
