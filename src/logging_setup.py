"""
Configuración de loggers. Idempotente.

Crea dos loggers dedicados:
- "inference": fallbacks de provider, errores transitorios del LLM remoto.
- "guardrail": fail-open del clasificador, categorías desconocidas.

Ambos van a stdout con un formato breve y NO propagan al root logger para
no contaminar otros sistemas (Gradio, Uvicorn) con su tráfico.
"""

import logging


def _build_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s %(message)s"))
    return handler


def _ensure_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(level)
        log.addHandler(_build_handler())
        log.propagate = False
    return log


def configure_loggers() -> None:
    """Configura los loggers `inference` y `guardrail`. Llamado por `app.py`."""
    _ensure_logger("inference")
    _ensure_logger("guardrail")
