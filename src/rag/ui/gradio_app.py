"""Layout Gradio del RAG personal.

Se limita a componer widgets y enlazar callbacks. Toda la lógica vive en
`handlers.py` y en la capa de `application`.
"""

from __future__ import annotations

from functools import partial

import gradio as gr

from ..bootstrap import AppServices
from ..config import settings
from .handlers import ingest_uploaded_file, respond


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
    ingest_fn = partial(ingest_uploaded_file, services)

    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITLE}\n\n{DESCRIPTION}")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Ajustes")
                top_k = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=settings.top_k,
                    step=1,
                    label="Fragmentos a recuperar (top_k)",
                )
                enable_reranker = gr.Checkbox(
                    value=settings.enable_reranker,
                    label="Reranker cruzado (más lento, más preciso)",
                )
                gr.Markdown("### Añadir documento")
                upload = gr.File(
                    label="Sube un PDF extra",
                    file_types=[".pdf"],
                    type="filepath",
                )
                upload_status = gr.Markdown()
                upload.change(
                    fn=ingest_fn,
                    inputs=[upload],
                    outputs=[upload_status],
                )

            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=respond_fn,
                    type="messages",
                    additional_inputs=[top_k, enable_reranker],
                    examples=[[q] for q in EXAMPLES],
                    cache_examples=False,
                )

        gr.Markdown(
            "---\n*Stack: Ollama + LangChain + FAISS + Gradio. "
            f"Modelo: `{settings.ollama_model}` · "
            f"Embeddings: `{settings.embed_model}`.*"
        )

    return demo
