"""Configuración centralizada de logging para la app.

Mantenemos un único punto de entrada para que los scripts y la app Gradio
compartan el mismo formato y nivel, evitando `print` dispersos.
"""

from __future__ import annotations

import logging

from .config import settings


_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_configured = False


def configure_logging(level: str | None = None) -> None:
    """Configura logging de stdlib. Idempotente: llamar varias veces no duplica handlers."""
    global _configured
    if _configured:
        return

    resolved = (level or settings.log_level or "INFO").upper()
    logging.basicConfig(level=resolved, format=_LOG_FORMAT)

    # Silenciar librerías ruidosas en INFO para que los logs propios se vean.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _configured = True
