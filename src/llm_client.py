"""
Cliente HF Inference Providers con fallback de provider.

Único módulo que toca el LLM remoto. El resto del pipeline llama a
`chat_completion_with_fallback(...)` y delega aquí la decisión de provider,
los reintentos y el routing del HF_TOKEN.

Estrategia: try primary (Cerebras, ~2000 t/s) y, ante cualquier excepción
(rate-limit, timeout, 5xx), fallback al secundario (Groq, ~500-800 t/s).
Una vez abierto un stream, los errores intermedios no se reintentan para
no duplicar texto al consumidor.
"""

import logging
import os

from huggingface_hub import InferenceClient

from src.config import AppConfig, get_config


_log = logging.getLogger("inference")
_primary: InferenceClient | None = None
_fallback: InferenceClient | None = None


def build_inference_client(provider: str, api_key: str | None = None) -> InferenceClient:
    """
    Construye un `InferenceClient` apuntando a HF Inference Providers.

    Crear el cliente NO dispara ninguna petición HTTP — la red solo se toca
    cuando alguna función invoca `chat.completions.create(...)`.
    """
    token = (api_key or os.getenv("HF_TOKEN") or "").strip()
    if not token:
        raise RuntimeError(
            "Falta HF_TOKEN en .env. Crea un token en "
            "https://huggingface.co/settings/tokens y añade HF_TOKEN=hf_... a tu .env."
        )
    return InferenceClient(provider=provider, api_key=token)


def init_clients(config: AppConfig | None = None) -> None:
    """
    Inicializa los singletons de cliente primario y fallback.

    Llamado por `app.py` durante el bootstrap. Idempotente: si los clientes
    ya están construidos no los reconstruye.
    """
    global _primary, _fallback
    if _primary is not None and _fallback is not None:
        return
    cfg = config if config is not None else get_config()
    _primary = build_inference_client(cfg.primary_provider, api_key=cfg.hf_token)
    _fallback = build_inference_client(cfg.fallback_provider, api_key=cfg.hf_token)


def _get_clients() -> tuple[InferenceClient, InferenceClient]:
    if _primary is None or _fallback is None:
        init_clients()
    assert _primary is not None and _fallback is not None
    return _primary, _fallback


def chat_completion_with_fallback(
    messages,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    stream: bool = False,
    primary: InferenceClient | None = None,
    fallback: InferenceClient | None = None,
):
    """
    Llama a `chat.completions.create()` contra el provider primario.
    Si falla con cualquier excepción reintenta con el fallback y loguea
    el motivo del fallo.

    Para streaming: el try/except envuelve solo la apertura del stream.
    Una vez abierto, los errores intermedios se propagan al consumidor.
    """
    cfg = get_config()
    p, f = (primary, fallback)
    if p is None or f is None:
        primary_client, fallback_client = _get_clients()
        p = p or primary_client
        f = f or fallback_client

    model_name = model or cfg.llm_model_name
    mt = max_tokens if max_tokens is not None else cfg.max_tokens
    temp = temperature if temperature is not None else cfg.temperature

    try:
        return p.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=mt,
            temperature=temp,
            stream=stream,
        )
    except Exception as exc:
        _log.warning(
            "primary provider %r failed, falling back to %r: %s",
            cfg.primary_provider, cfg.fallback_provider, exc,
        )
        return f.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=mt,
            temperature=temp,
            stream=stream,
        )


def extract_message_text(message) -> str:
    """
    Normaliza la respuesta del LLM a string. Algunos providers devuelven
    `str` y otros una lista de bloques `{"type": "text", "text": "..."}`.
    """
    if isinstance(message, list):
        return "".join(
            block.get("text", "")
            for block in message
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return str(message).strip()
