"""
Muro de configuración. Único módulo que toca .env y os.getenv().

El resto del código consume `AppConfig` por inyección o lee el singleton
`CONFIG` (resultado de `load_config()`). Mantener todo el acceso a entorno
aquí asegura que un único cambio basta para auditar cómo se inyecta la
configuración en el sistema.

Variables esperadas en .env:
    HF_TOKEN=hf_...
    CONTACT_EMAIL=tu@email.com
    CONTACT_LINKEDIN_URL=https://www.linkedin.com/in/tu-perfil/
    GRADIO_SERVER_NAME=0.0.0.0   (opcional)
    PORT=7860                    (opcional)
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """Configuración global del proyecto, inmutable tras `load_config()`."""

    project_root: Path
    profile_path: Path

    # Ingesta y retrieval
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int

    # LLM
    llm_model_name: str
    primary_provider: str
    fallback_provider: str
    max_tokens: int
    temperature: float
    hf_token: str

    # Datos de contacto (para el footer)
    contact_email: str
    contact_linkedin_url: str

    # Servidor Gradio
    server_name: str
    server_port: int


_CONFIG: AppConfig | None = None


def _detect_project_root() -> Path:
    """
    Devuelve la raíz del proyecto. Si la cwd es `notebooks/`, sube un nivel.
    """
    cwd = Path.cwd()
    if cwd.name == "notebooks":
        return cwd.parent
    return cwd


def load_config() -> AppConfig:
    """
    Lee .env y construye una instancia inmutable de `AppConfig`.

    Idempotente: una vez construida, devuelve la misma instancia. El singleton
    se expone como `CONFIG` para los módulos que solo necesitan defaults.
    """
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    project_root = _detect_project_root()
    load_dotenv(project_root / ".env")

    config = AppConfig(
        project_root=project_root,
        profile_path=project_root / "data" / "profile" / "profile.md",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=100,
        retrieval_k=5,
        llm_model_name="meta-llama/Llama-3.3-70B-Instruct",
        primary_provider="cerebras",
        fallback_provider="groq",
        max_tokens=300,
        temperature=0.2,
        hf_token=(os.getenv("HF_TOKEN") or "").strip(),
        contact_email=(os.getenv("CONTACT_EMAIL") or "").strip(),
        contact_linkedin_url=(os.getenv("CONTACT_LINKEDIN_URL") or "").strip(),
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
    )
    _CONFIG = config
    return config


def get_config() -> AppConfig:
    """Devuelve el singleton de configuración. Lo construye si no existe."""
    return _CONFIG if _CONFIG is not None else load_config()
