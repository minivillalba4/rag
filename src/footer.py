"""
Footer de contacto que se añade al final de toda respuesta.

Se aplica a TODAS las salidas del asistente (respuestas legítimas y
mensajes de rechazo del guardrail) sin excepción, para garantizar que
el reclutador siempre tenga email y LinkedIn a mano.

Vive en su propio módulo porque lo consumen `generation`, `guardrail`
y `ui/app`. Tenerlo aparte evita ciclos de import.
"""

from src.config import get_config


def append_contact_footer(text: str) -> str:
    """
    Añade un bloque fijo de contacto al final de un texto.

    Si ninguna de las dos variables (CONTACT_EMAIL, CONTACT_LINKEDIN_URL)
    está configurada en .env, devuelve el texto sin modificar (degradación
    silenciosa).
    """
    config = get_config()
    body = text.rstrip()

    lines = []
    if config.contact_email:
        lines.append(f"- Email: [{config.contact_email}](mailto:{config.contact_email})")
    if config.contact_linkedin_url:
        lines.append(f"- LinkedIn: [mi perfil]({config.contact_linkedin_url})")
    if not lines:
        return body
    return body + "\n\n---\n**¿Hablamos?**\n" + "\n".join(lines)
