"""
Avatares del chat.

- Foto real para Ismael (`data/1758611341302.jpeg`).
- Marca minimal SVG para el visitante (círculo tinta con punto óxido).

Importante: en Gradio 6, `avatar_images` espera RUTAS A FICHEROS. Las
data URIs no funcionan: el `open()` interno falla con `OSError [Errno 22]`.
Por eso escribimos el SVG a un fichero real.
"""

from pathlib import Path


_VISITOR_AVATAR_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<circle cx='32' cy='32' r='32' fill='#1f1b14'/>"
    "<circle cx='32' cy='32' r='2.6' fill='#c44d34'/>"
    "</svg>"
)


def build_avatars(project_root: Path) -> tuple[str, str]:
    """
    Devuelve `(visitor_path, ismael_path)` como strings absolutos.

    Si el SVG del visitante no existe, lo escribe en `data/visitor_avatar.svg`.
    """
    data_dir = project_root / "data"
    visitor_path = data_dir / "visitor_avatar.svg"
    ismael_path = data_dir / "1758611341302.jpeg"

    visitor_path.parent.mkdir(parents=True, exist_ok=True)
    if not visitor_path.exists():
        visitor_path.write_text(_VISITOR_AVATAR_SVG, encoding="utf-8")

    return str(visitor_path), str(ismael_path)
