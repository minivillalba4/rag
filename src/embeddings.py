"""
Construcción del modelo de embeddings.

Wrapper fino sobre `HuggingFaceEmbeddings`. La carga del modelo es costosa
(descarga + carga en memoria), así que se hace una sola vez al construir
el pipeline.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_config


def build_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    """Crea el modelo de embeddings. Por defecto usa el de `AppConfig`."""
    config = get_config()
    name = model_name if model_name is not None else config.embedding_model
    return HuggingFaceEmbeddings(model_name=name)
