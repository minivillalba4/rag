"""Interfaz Gradio del RAG personal."""

from __future__ import annotations

import shutil
from pathlib import Path

import gradio as gr
from langchain_ollama import ChatOllama

from .chain import build_rag_chain
from .config import settings
from .ingest import add_documents_to_index
from .loaders import load_documents
from .vectorstore import load_or_build_vectorstore


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


# --- Recursos singleton (se cargan una vez al arrancar) ------------------

_vectorstore = load_or_build_vectorstore()
_llm = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    temperature=0.2,
)


def _format_sources(docs) -> str:
    if not docs:
        return ""
    lines = ["\n\n---\n**Fuentes**"]
    seen: set[tuple[str, int | None]] = set()
    for d in docs:
        src = d.metadata.get("source", "desconocido")
        page = d.metadata.get("page")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        page_str = f" (p.{page + 1})" if page is not None else ""
        lines.append(f"- `{src}`{page_str}")
    return "\n".join(lines)


async def respond(message: str, history: list[dict], top_k: int):
    """Async generator: streamea tokens y añade citas al final."""
    bundle = build_rag_chain(vectorstore=_vectorstore, llm=_llm, top_k=int(top_k))

    docs = await bundle.retriever.ainvoke(message)

    partial = ""
    async for chunk in bundle.chain.astream(message):
        partial += chunk
        yield partial

    yield partial + _format_sources(docs)


def ingest_uploaded_file(file_obj) -> str:
    """Copia el PDF subido a data/sources y lo añade al índice."""
    if file_obj is None:
        return ""

    src_path = Path(file_obj.name)
    dest = settings.data_dir / src_path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path, dest)

    new_docs = load_documents(settings.data_dir)
    new_docs = [d for d in new_docs if d.metadata.get("source") == src_path.name]
    add_documents_to_index(new_docs)

    global _vectorstore
    _vectorstore = load_or_build_vectorstore()

    return f"Añadido **{src_path.name}** al índice ({len(new_docs)} fragmentos)."


def build_demo() -> gr.Blocks:
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
                gr.Markdown("### Añadir documento")
                upload = gr.File(
                    label="Sube un PDF extra",
                    file_types=[".pdf"],
                    type="filepath",
                )
                upload_status = gr.Markdown()
                upload.change(
                    fn=ingest_uploaded_file,
                    inputs=[upload],
                    outputs=[upload_status],
                )

            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=respond,
                    type="messages",
                    additional_inputs=[top_k],
                    examples=[[q] for q in EXAMPLES],
                    cache_examples=False,
                )

        gr.Markdown(
            "---\n*Stack: Ollama + LangChain + FAISS + Gradio. "
            f"Modelo: `{settings.ollama_model}` · "
            f"Embeddings: `{settings.embed_model}`.*"
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
    )


if __name__ == "__main__":
    main()
