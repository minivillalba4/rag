from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Proveedores: "ollama" (local, por defecto) o "groq" (API gratuita).
    llm_provider: str = "ollama"
    # Proveedores de embeddings: "ollama" o "huggingface" (modelo local CPU).
    embed_provider: str = "ollama"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    embed_model: str = "nomic-embed-text"

    # Groq (free tier en https://console.groq.com/keys)
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    # HuggingFace embeddings locales (CPU)
    hf_embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    top_k: int = 4
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Memoria conversacional: últimos N turnos considerados al reescribir.
    history_window: int = 6

    # Reranker cruzado opcional (requiere `sentence-transformers`).
    enable_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    retriever_fetch_k: int = 20

    data_dir: Path = ROOT_DIR / "data" / "sources"
    index_dir: Path = ROOT_DIR / "data" / "index"

    gradio_server_name: str = "0.0.0.0"
    gradio_server_port: int = 7860

    log_level: str = "INFO"

    # Evaluación: modelo Ollama usado como juez RAGAS. Si es None, se usa
    # el mismo `ollama_model`. Para mejor calidad de juicio, apuntar a un
    # modelo mayor (ej. "llama3.1:70b").
    eval_judge_model: str | None = None

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
