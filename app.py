"""
Punto de entrada único del proyecto RAG portfolio.

Arranque en orden:
    1. load_config()        — lee .env y construye el singleton AppConfig.
    2. configure_loggers()  — loggers `inference` y `guardrail`.
    3. init_clients(config) — clientes HF Cerebras (primario) + Groq (fallback).
    4. build_rag_pipeline   — ingesta + chunking + embeddings + FAISS.
    5. launch_demo(config)  — Gradio ChatInterface con streaming.

Ejecución:
    python app.py
"""

import logging

from src.config import load_config
from src.llm_client import init_clients
from src.logging_setup import configure_loggers
from src.pipeline import build_rag_pipeline
from src.ui.app import launch_demo


def main() -> None:
    config = load_config()
    configure_loggers()
    init_clients(config)

    log = logging.getLogger("inference")
    log.info("Indexando documento de perfil: %s", config.profile_path.name)
    pipeline = build_rag_pipeline(config)
    log.info(
        "Pipeline listo. Chunks=%d k=%d modelo=%s providers=%s→%s",
        pipeline.vectorstore.index.ntotal,
        config.retrieval_k,
        config.llm_model_name,
        config.primary_provider,
        config.fallback_provider,
    )

    launch_demo(config)


if __name__ == "__main__":
    main()
