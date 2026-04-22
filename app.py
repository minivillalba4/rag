"""Entry point para HuggingFace Spaces y Render.

HF Spaces espera `app.py` en la raíz con una variable `demo` que sea un
`gr.Blocks`. Aquí construimos los servicios y exponemos tanto `demo`
(para el runtime de Spaces) como `main()` (para `python app.py`).
"""

from __future__ import annotations

import logging

import gradio as gr

from src.rag.bootstrap import build_app_services
from src.rag.config import settings
from src.rag.logging_config import configure_logging
from src.rag.ui.gradio_app import build_demo


configure_logging()
logger = logging.getLogger(__name__)


def _seed_corpus_if_missing() -> None:
    """Garantiza que data_dir existe y contiene al menos un documento.

    En HuggingFace Spaces (Gradio SDK) los directorios vacíos o con solo
    `.gitkeep` no siempre se materializan; sin este seed, `load_documents`
    crashea en frío. Generamos el CV por defecto si el corpus está vacío.
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    has_any = any(
        f
        for pat in ("*.pdf", "*.md", "*.txt")
        for f in settings.data_dir.glob(pat)
    )
    if has_any:
        return
    logger.info("Corpus vacío en %s; generando CV por defecto", settings.data_dir)
    from scripts.generate_cv_pdf import build_pdf

    build_pdf(settings.data_dir / "cv_candidato.pdf")

logger.info(
    "Arrancando app (llm=%s, embed=%s, top_k=%d, reranker=%s)",
    settings.llm_provider,
    settings.embed_provider,
    settings.top_k,
    settings.enable_reranker,
)

_seed_corpus_if_missing()
services = build_app_services()
demo = build_demo(services)


def main() -> None:
    demo.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
