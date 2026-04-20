from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    embed_model: str = "nomic-embed-text"

    top_k: int = 4
    chunk_size: int = 500
    chunk_overlap: int = 50

    data_dir: Path = ROOT_DIR / "data" / "sources"
    index_dir: Path = ROOT_DIR / "data" / "index"

    gradio_server_name: str = "0.0.0.0"
    gradio_server_port: int = 7860

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
