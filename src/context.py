"""
Formateo del contexto que se inyecta en el prompt del LLM.

Toma la lista normalizada de fuentes y produce un bloque de texto
con índices `[0]`, `[1]`, ... que el SYSTEM_PROMPT espera ver.
"""


def build_context(sources) -> str:
    """Concatena los chunks en un bloque `[i] content` separado por líneas en blanco."""
    return "\n\n".join(
        f"[{index}] {source['content']}" for index, source in enumerate(sources)
    )
