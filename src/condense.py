"""
Reformulación de la pregunta de seguimiento como pregunta autónoma.

Le da al retrieval una pregunta completamente entendible por sí sola
("¿y eso por qué?" → "¿por qué dejaste tu último trabajo?"), de modo que
los embeddings recuperen los chunks correctos sin depender del historial.

Si no hay historial, devuelve la pregunta tal cual y se ahorra la
llamada al LLM.
"""

from src.llm_client import chat_completion_with_fallback, extract_message_text
from src.prompts import CONDENSE_SYSTEM_PROMPT


def format_history_for_condense(history) -> str:
    """Formatea el historial como `Usuario: ... / Asistente: ...` para el LLM."""
    lines = []
    for turn in history:
        role = "Usuario" if turn["role"] == "user" else "Asistente"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def condense_question(history, question: str, model_name: str | None = None) -> str:
    """
    Reformula la pregunta de seguimiento como autónoma. Devuelve la pregunta
    tal cual si el historial está vacío.
    """
    if not history:
        return question

    user_input = (
        f"HISTORIAL:\n{format_history_for_condense(history)}\n\n"
        f"PREGUNTA DE SEGUIMIENTO:\n{question}\n\n"
        "PREGUNTA INDEPENDIENTE:"
    )

    response = chat_completion_with_fallback(
        messages=[
            {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        model=model_name,
        max_tokens=200,
        temperature=0.0,
    )

    return extract_message_text(response.choices[0].message.content)
