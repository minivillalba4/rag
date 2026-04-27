"""
Adaptador Gradio del chatbot.

`gradio_chat(message, history)` es un generador que:
1. Clasifica la pregunta con el guardrail.
2. Si no es "allow", emite el mensaje fijo de refusal con footer.
3. Si es "allow", reformula la pregunta (condense), recupera contexto
   y stremea la generación token a token.

`build_demo(config)` construye el `gr.ChatInterface` con tema y CSS.
`launch_demo(config)` lo arranca en el puerto configurado.

Nota: la historia que recibe `gradio_chat` viene del `ChatInterface` de
Gradio (que la gestiona en el cliente). El pipeline `ConversationalRagPipeline`
mantiene su propia historia internamente para uso programático; aquí
respetamos la historia del lado UI para que ambos modos coexistan sin
duplicar estado.
"""

import logging

import gradio as gr

from src.condense import condense_question
from src.config import AppConfig, get_config
from src.context import build_context
from src.footer import append_contact_footer
from src.generation import build_user_input, stream_answer_with_history
from src.guardrail import REFUSAL_MESSAGES, classify_intent
from src.pipeline import get_cached_pipeline
from src.retriever import normalize_sources, retrieve_documents
from src.ui.avatars import build_avatars
from src.ui.theme import CUSTOM_CSS, build_theme


_log = logging.getLogger("inference")


EXAMPLE_QUESTIONS = [
    "Háblame de ti",
    "¿Por qué deberíamos contratarte?",
]


# Mensaje que se muestra al usuario si TODO el stack de providers
# (Cerebras + Groq) falla, o si cualquier paso del pipeline lanza una
# excepción no controlada. El footer de contacto lo añade
# `append_contact_footer`, así que el reclutador siempre tiene una vía
# alternativa para llegar a Ismael.
TECH_ERROR_MESSAGE = (
    "Tengo un problema técnico temporal y no puedo responder ahora mismo. "
    "Prueba a reformular la pregunta en un momento, o contáctame "
    "directamente por los canales de abajo."
)


def gradio_chat(message: str, history):
    """
    Generador con streaming. En cada chunk del LLM hace `yield` del texto
    parcial acumulado y Gradio actualiza la burbuja al vuelo.

    Si TODO el stack falla o cualquier paso lanza una excepción inesperada,
    se captura aquí: la traza se loguea con `exc_info=True` y al usuario se
    le muestra un mensaje amable con el footer (sin filtrar stacktrace).
    """
    if not message or not message.strip():
        raise gr.Error("Escribe una pregunta.")

    pregunta = message.strip()
    pipeline = get_cached_pipeline()
    config = get_config()

    try:
        intent = classify_intent(pregunta)
        if intent["category"] != "allow":
            yield append_contact_footer(REFUSAL_MESSAGES[intent["category"]])
            return

        standalone = condense_question(history, pregunta)
        docs = retrieve_documents(pipeline.vectorstore, standalone, k=config.retrieval_k)
        sources = normalize_sources(docs)
        context = build_context(sources)

        if not context.strip():
            yield append_contact_footer(
                "No tengo información suficiente en el perfil para responder a esa pregunta."
            )
            return

        # Reaprovechamos `build_user_input` para validar que el formato
        # del mensaje user es el mismo que el resto del pipeline espera.
        _ = build_user_input(context, pregunta)

        for partial in stream_answer_with_history(context, pregunta, history):
            yield partial

    except Exception as exc:
        _log.error("gradio_chat fallo no controlado: %s", exc, exc_info=True)
        yield append_contact_footer(TECH_ERROR_MESSAGE)


def _build_chatbot(config: AppConfig) -> gr.Chatbot:
    visitor_path, ismael_path = build_avatars(config.project_root)
    return gr.Chatbot(
        height=720,
        show_label=False,
        buttons=["copy"],
        avatar_images=(visitor_path, ismael_path),
        placeholder=(
            "<div style=\"font-family: 'Fraunces', Georgia, serif; font-size: 1.1rem; "
            "color: #5d503d; line-height: 1.5; padding: 2.5rem 1.5rem; text-align: center;\">"
            "<div style=\"font-family: 'JetBrains Mono', ui-monospace, monospace; "
            "font-size: 0.7rem; letter-spacing: 0.22em; text-transform: uppercase; "
            "color: #c44d34; margin-bottom: 1rem;\">— Empezar conversación —</div>"
            "Pregúntame por mi experiencia, los proyectos en los que he trabajado "
            "o el stack técnico que domino."
            "</div>"
        ),
    )


def _build_textbox() -> gr.Textbox:
    return gr.Textbox(
        placeholder="Escribe tu pregunta…",
        container=False,
        show_label=False,
        autofocus=True,
        lines=1,
        max_lines=4,
    )


def build_demo(config: AppConfig | None = None) -> gr.ChatInterface:
    """Construye el `ChatInterface` con tema, CSS, avatares y placeholder."""
    cfg = config if config is not None else get_config()
    return gr.ChatInterface(
        fn=gradio_chat,
        chatbot=_build_chatbot(cfg),
        textbox=_build_textbox(),
        title="Hablemos de mi perfil.",
        description=(
            "Asistente conversacional sobre mi experiencia, formación y proyectos. "
            "Recupera contexto de mi perfil con RAG, mantiene el hilo entre turnos "
            "y filtra peticiones fuera de alcance."
        ),
        examples=EXAMPLE_QUESTIONS,
    )


def launch_demo(config: AppConfig | None = None) -> None:
    """Cierra cualquier servidor previo, encola y arranca el demo."""
    cfg = config if config is not None else get_config()
    gr.close_all()
    demo = build_demo(cfg)
    demo.queue()
    demo.launch(
        theme=build_theme(),
        css=CUSTOM_CSS,
        server_name=cfg.server_name,
        server_port=cfg.server_port,
        share=False,
    )
