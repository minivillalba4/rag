"""
Generación de respuestas del asistente.

Tres variantes:
- `generate_answer`: single-turn, sin historia (Q&A simple).
- `generate_answer_with_history`: multi-turn, devuelve la respuesta completa.
- `stream_answer_with_history`: multi-turn, generador que produce texto
  parcial token a token (consumido por la UI Gradio).

Todas usan `chat_completion_with_fallback`, que aplica el fallback de
provider HF si el primario falla.
"""

from src.footer import append_contact_footer
from src.llm_client import chat_completion_with_fallback, extract_message_text
from src.prompts import SYSTEM_PROMPT


def build_user_input(context: str, question: str) -> str:
    """Compone el bloque CONTEXTO + PREGUNTA que se inyecta como mensaje user."""
    return f"""
CONTEXTO:
{context}

PREGUNTA:
{question}
""".strip()


def _no_context_message() -> str:
    return append_contact_footer(
        "No tengo información suficiente en el perfil para responder a esa pregunta."
    )


def generate_answer(context: str, question: str, model_name: str | None = None) -> str:
    """Versión single-turn (sin historial)."""
    if not context.strip():
        return _no_context_message()

    try:
        response = chat_completion_with_fallback(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_input(context, question)},
            ],
            model=model_name,
        )
    except Exception as exc:
        raise RuntimeError(f"Error al llamar al LLM: {exc}") from exc

    return append_contact_footer(extract_message_text(response.choices[0].message.content))


def _build_messages_with_history(context: str, question: str, history) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": build_user_input(context, question)})
    return messages


def generate_answer_with_history(
    context: str,
    question: str,
    history,
    model_name: str | None = None,
) -> str:
    """Versión multi-turn no streaming. Devuelve la respuesta completa."""
    if not context.strip():
        return _no_context_message()

    try:
        response = chat_completion_with_fallback(
            messages=_build_messages_with_history(context, question, history),
            model=model_name,
        )
    except Exception as exc:
        raise RuntimeError(f"Error al llamar al LLM: {exc}") from exc

    return append_contact_footer(extract_message_text(response.choices[0].message.content))


def stream_answer_with_history(
    context: str,
    question: str,
    history,
    model_name: str | None = None,
):
    """
    Generador con streaming. En cada iteración devuelve el texto parcial
    acumulado (no solo el delta). El último yield contiene el texto
    completo + footer de contacto.

    Si no hay contexto, hace un único yield con el mensaje de fallback.
    """
    if not context.strip():
        yield _no_context_message()
        return

    stream = chat_completion_with_fallback(
        messages=_build_messages_with_history(context, question, history),
        model=model_name,
        stream=True,
    )

    partial = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            partial += delta
            yield partial

    yield append_contact_footer(partial)
