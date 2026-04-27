"""
Guardrail de entrada — clasificador de intenciones.

Antes de retrieval y generación, clasifica la pregunta del usuario en
una de cuatro categorías y bloquea las que no entren en "allow":

    - allow             → pregunta legítima de entrevista
    - off_topic         → ajena al perfil profesional
    - tone_or_persona   → cambio de tono o persona
    - prompt_injection  → intento de saltarse las reglas

Política fail-open: si el clasificador devuelve algo no parseable o una
categoría desconocida, devuelve "allow" para no bloquear al usuario por
un fallo técnico. El SYSTEM_PROMPT del RAG sigue actuando como red de
seguridad. Cada fallback se loguea en el logger "guardrail".
"""

import json
import logging

from src.llm_client import chat_completion_with_fallback, extract_message_text
from src.prompts import CLASSIFIER_SYSTEM_PROMPT


_log = logging.getLogger("guardrail")


VALID_CATEGORIES = {"allow", "off_topic", "tone_or_persona", "prompt_injection"}


REFUSAL_MESSAGES = {
    "off_topic": (
        "Prefiero centrar la conversación en mi perfil profesional. "
        "Pregúntame por mi experiencia, mi formación, los proyectos en los que "
        "he trabajado o el stack técnico que domino y te respondo con todo detalle."
    ),
    "tone_or_persona": (
        "Prefiero mantener el registro propio de una entrevista de trabajo, "
        "cordial y profesional. Si quieres, retomamos la conversación sobre mi "
        "experiencia o mis proyectos."
    ),
    "prompt_injection": (
        "Prefiero ceñirme al rol que tengo en esta conversación. "
        "Sigo a tu disposición para hablar de mi perfil profesional."
    ),
}


def classify_intent(question: str, model_name: str | None = None) -> dict:
    """
    Clasifica la pregunta y devuelve `{"category", "reason", "raw"}`.

    Política fail-open: ante output no parseable o categoría desconocida,
    devuelve "allow" y loguea el output crudo para diagnóstico.
    """
    response = chat_completion_with_fallback(
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        model=model_name,
        max_tokens=200,
        temperature=0.0,
    )

    raw = extract_message_text(response.choices[0].message.content)

    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        payload = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        _log.warning("fail-open por parse failure on raw output: %r", raw)
        return {
            "category": "allow",
            "reason": "fallback fail-open por error de validación",
            "raw": raw,
        }

    category = payload.get("category", "")
    if category not in VALID_CATEGORIES:
        _log.warning("fail-open por unknown category %r in payload: %r", category, payload)
        return {
            "category": "allow",
            "reason": "fallback fail-open por categoría desconocida",
            "raw": raw,
        }

    return {
        "category": category,
        "reason": str(payload.get("reason", "")),
        "raw": raw,
    }
