"""Layout Gradio del RAG personal.

Se limita a componer widgets y enlazar callbacks. Toda la lógica vive en
`handlers.py` y en la capa de `application`.
"""

from __future__ import annotations

from functools import partial

import gradio as gr

from ..bootstrap import AppServices
from .handlers import respond


TITLE = "RAG sobre mi CV — asistente para recruiters"
DESCRIPTION = (
    "Pregunta lo que quieras saber sobre mi experiencia profesional. "
    "El asistente responde únicamente con la información de los documentos "
    "(CV, proyectos) y cita sus fuentes."
)

EXAMPLES = [
    "¿Qué experiencia tiene en LLMs y RAG?",
    "Resume los proyectos más relevantes de NLP.",
    "¿Qué stack técnico domina en Python?",
    "¿Tiene experiencia con Docker y despliegue en la nube?",
    "¿Habla inglés? ¿Qué nivel?",
]


def build_demo(services: AppServices) -> gr.Blocks:
    respond_fn = partial(respond, services)

    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\n\n{DESCRIPTION}")
        gr.ChatInterface(
            fn=respond_fn,
            examples=[[q] for q in EXAMPLES],
            cache_examples=False,
        )

    return demo
