"""Entry point de la aplicación.

Responsabilidad única: leer config, instanciar servicios vía bootstrap,
construir la UI y lanzarla. Nada más.
"""

from __future__ import annotations

import logging

import gradio as gr

from .bootstrap import build_app_services
from .config import settings
from .logging_config import configure_logging
from .ui.gradio_app import build_demo


logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    logger.info(
        "Arrancando app (modelo=%s, embed=%s, top_k=%d, reranker=%s)",
        settings.ollama_model,
        settings.embed_model,
        settings.top_k,
        settings.enable_reranker,
    )
    services = build_app_services()
    demo = build_demo(services)
    demo.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
