"""Entry point para HuggingFace Spaces y Render.

HF Spaces espera `app.py` en la raíz con una variable `demo` que sea un
`gr.Blocks`. Aquí construimos los servicios y exponemos tanto `demo`
(para el runtime de Spaces) como `main()` (para `python app.py`).
"""

from __future__ import annotations

import logging

from src.rag.bootstrap import build_app_services
from src.rag.config import settings
from src.rag.logging_config import configure_logging
from src.rag.ui.gradio_app import build_demo


configure_logging()
logger = logging.getLogger(__name__)

logger.info(
    "Arrancando app (llm=%s, embed=%s, top_k=%d, reranker=%s)",
    settings.llm_provider,
    settings.embed_provider,
    settings.top_k,
    settings.enable_reranker,
)

services = build_app_services()
demo = build_demo(services)


def main() -> None:
    demo.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
    )


if __name__ == "__main__":
    main()
