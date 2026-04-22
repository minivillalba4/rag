"""Composition root de la aplicación.

Único sitio donde se instancian los recursos de larga vida (vectorstore,
LLM) y donde vive la caché de bundles RAG por configuración. La UI, los
scripts y los tests reciben este objeto inyectado en lugar de leer
singletons globales.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from .application.ask import RagBundle, build_ask_service
from .config import settings
from .infrastructure.faiss_store import load_or_build_vectorstore


logger = logging.getLogger(__name__)


def _build_llm() -> BaseChatModel:
    provider = settings.llm_provider.lower()
    if provider == "groq":
        from .infrastructure.groq_llm import build_groq_llm

        logger.info("LLM: Groq (%s)", settings.groq_model)
        return build_groq_llm()
    if provider == "ollama":
        from .infrastructure.ollama_llm import build_chat_llm

        logger.info("LLM: Ollama (%s)", settings.ollama_model)
        return build_chat_llm()
    raise ValueError(f"LLM_PROVIDER desconocido: {provider!r}")


@dataclass
class AppServices:
    """Recursos compartidos entre requests y caché de bundles RAG."""

    vectorstore: FAISS
    llm: BaseChatModel
    _bundle_cache: dict[tuple[int, bool], RagBundle] = field(default_factory=dict)

    def get_bundle(self, top_k: int, enable_reranker: bool) -> RagBundle:
        key = (top_k, enable_reranker)
        bundle = self._bundle_cache.get(key)
        if bundle is None:
            bundle = build_ask_service(
                vectorstore=self.vectorstore,
                llm=self.llm,
                top_k=top_k,
                enable_reranker=enable_reranker,
            )
            self._bundle_cache[key] = bundle
        return bundle

    def reload_vectorstore(self) -> None:
        """Recarga el índice desde disco e invalida bundles cacheados.

        Los bundles cachean retrievers ligados al vectorstore anterior; tras
        ingerir documentos nuevos hay que descartarlos.
        """
        self.vectorstore = load_or_build_vectorstore()
        self._bundle_cache.clear()
        logger.info("Vectorstore recargado; caché de bundles invalidada")


def build_app_services() -> AppServices:
    """Instancia los adapters concretos y devuelve el contenedor."""
    return AppServices(
        vectorstore=load_or_build_vectorstore(),
        llm=_build_llm(),
    )
