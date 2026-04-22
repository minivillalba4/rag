"""Genera el CV en PDF que sirve de corpus al RAG.

El contenido está alineado con las preguntas de tests/eval/golden_dataset.jsonl
para que la evaluación con RAGAS produzca métricas significativas.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import reportlab
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT_DIR / "data" / "sources" / "cv_candidato.pdf"

FONT_REGULAR = "Vera"
FONT_BOLD = "Vera-Bold"


def _register_fonts() -> None:
    fonts_dir = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(fonts_dir / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(fonts_dir / "VeraBd.ttf")))


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    body = ParagraphStyle(
        "Body",
        parent=base,
        fontName=FONT_REGULAR,
        fontSize=10,
        leading=13,
        spaceAfter=4,
    )
    title = ParagraphStyle(
        "Title",
        parent=body,
        fontName=FONT_BOLD,
        fontSize=18,
        leading=22,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=body,
        fontName=FONT_BOLD,
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=4,
        textColor="#1C3A5E",
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=14,
        bulletIndent=0,
        spaceAfter=2,
    )
    return {"title": title, "h2": h2, "body": body, "bullet": bullet}


def _content(styles: dict[str, ParagraphStyle]) -> list:
    s = styles
    story: list = []

    story += [
        Paragraph("Carlos Candidato — Ingeniero de IA / Backend", s["title"]),
        Paragraph(
            "Madrid · carlos.candidato@ejemplo.es · +34 600 000 000 · "
            "github.com/ejemplo-candidato",
            s["body"],
        ),
        Spacer(1, 0.3 * cm),
    ]

    story += [
        Paragraph("Perfil profesional", s["h2"]),
        Paragraph(
            "Ingeniero con siete años de experiencia construyendo sistemas backend "
            "y, los últimos tres, aplicaciones basadas en LLMs. Especializado en "
            "RAG en producción, arquitecturas hexagonales y evaluación continua "
            "con RAGAS. Valoro la simplicidad, los tests deterministas y la "
            "observabilidad desde el primer commit.",
            s["body"],
        ),
    ]

    story += [
        Paragraph("Experiencia profesional", s["h2"]),
        Paragraph(
            "<b>Empresa Ejemplo S.L.</b> — Ingeniero de IA (2022 — actualidad). "
            "Diseñó e implementó un sistema RAG en producción sobre documentación "
            "interna usando LangChain, FAISS y Ollama. Lideró la migración del "
            "stack NLP, introdujo evaluación automática con RAGAS y estableció "
            "el pipeline de CI/CD sobre GitHub Actions.",
            s["body"],
        ),
        Spacer(1, 0.15 * cm),
        Paragraph(
            "<b>Consultora Tech</b> — Desarrollador Backend (2019 — 2022). "
            "Responsable de APIs REST en FastAPI con PostgreSQL y Redis, "
            "integraciones con proveedores externos (pasarelas de pago, "
            "servicios de mensajería) y despliegue continuo en AWS.",
            s["body"],
        ),
    ]

    story += [
        Paragraph("Stack técnico", s["h2"]),
        Paragraph(
            "<b>Backend:</b> FastAPI, Flask, PostgreSQL, Redis, Docker.",
            s["bullet"],
        ),
        Paragraph(
            "<b>NLP / IA:</b> LangChain, HuggingFace, PyTorch, "
            "sentence-transformers, FAISS, Ollama, RAG y fine-tuning ligero "
            "con LoRA.",
            s["bullet"],
        ),
        Paragraph(
            "<b>Cloud / DevOps:</b> AWS (EC2, S3, Lambda), Docker Compose y "
            "CI/CD en GitHub Actions.",
            s["bullet"],
        ),
    ]

    story += [
        Paragraph("Proyectos destacados", s["h2"]),
        Paragraph(
            "<b>RAG personal sobre CV</b> — asistente conversacional con "
            "Ollama, LangChain y Gradio que responde preguntas sobre su "
            "propia trayectoria y cita las fuentes.",
            s["bullet"],
        ),
        Paragraph(
            "<b>Clasificador de tickets de soporte</b> — modelo que obtuvo "
            "un F1 de 0.89 clasificando tickets en 12 categorías, y para el "
            "que se redujo un 40 % la latencia del pipeline de embeddings "
            "mediante chunking adaptativo y caching selectivo. Ambos "
            "resultados se consiguieron en el mismo proyecto.",
            s["bullet"],
        ),
        Paragraph(
            "<b>Dashboard analítico</b> — cuadro de mando interno en "
            "Streamlit con backend DuckDB para análisis ad-hoc sobre "
            "conjuntos de varios millones de filas sin infraestructura extra.",
            s["bullet"],
        ),
    ]

    story += [
        Paragraph("Formación", s["h2"]),
        Paragraph(
            "Grado y Máster en Ingeniería Informática. Formación continua en "
            "LLM Engineering (DeepLearning.AI) y Advanced RAG (LlamaIndex).",
            s["body"],
        ),
    ]

    story += [
        Paragraph("Idiomas", s["h2"]),
        Paragraph(
            "Español nativo. Inglés C1, apto para entornos técnicos y "
            "reuniones internacionales.",
            s["body"],
        ),
    ]

    story += [
        Paragraph("Buenas prácticas que aplico", s["h2"]),
        Paragraph(
            "Arquitectura hexagonal estricta: separación de dominio, "
            "aplicación e infraestructura para que los adapters (Ollama, "
            "FAISS, Gradio) sean intercambiables sin tocar la lógica.",
            s["bullet"],
        ),
        Paragraph(
            "Tests con pytest y fixtures reutilizables, complementados con "
            "un dataset dorado evaluado en CI mediante RAGAS "
            "(faithfulness, answer-relevancy, context-precision, "
            "context-recall).",
            s["bullet"],
        ),
        Paragraph(
            "Linting con ruff, type-checking con mypy y pre-commit hooks "
            "para evitar regresiones de estilo o tipos en el repositorio.",
            s["bullet"],
        ),
        Paragraph(
            "Observabilidad de serie: logging estructurado, niveles "
            "configurables por entorno y trazas mínimas en cada request "
            "del RAG para depurar retrieval vs. generación.",
            s["bullet"],
        ),
        Paragraph(
            "Gestión reproducible de dependencias con pyproject.toml y "
            "lockfile; separación estricta de secretos (.env, nunca en "
            "el repositorio) y variables por entorno.",
            s["bullet"],
        ),
        Paragraph(
            "Contenedorización con Docker y docker-compose para ofrecer "
            "una experiencia «clona y levanta» a cualquier revisor, con "
            "volúmenes persistentes para índices y modelos.",
            s["bullet"],
        ),
        Paragraph(
            "Prompt engineering versionado: los prompts viven en módulos "
            "separados del código que los consume y se someten al mismo "
            "control de cambios que cualquier otro componente.",
            s["bullet"],
        ),
        Paragraph(
            "CI/CD automatizado en GitHub Actions con jobs separados para "
            "tests unitarios, evaluación RAGAS sobre el dataset dorado y "
            "build de la imagen Docker.",
            s["bullet"],
        ),
        Paragraph(
            "Evaluación continua de LLMs: toda mejora se contrasta contra "
            "métricas objetivas antes de considerarse lista para producción; "
            "no se acepta «parece que responde mejor» sin números.",
            s["bullet"],
        ),
    ]

    return story


def build_pdf(output: Path) -> Path:
    _register_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="CV — Carlos Candidato",
        author="Carlos Candidato",
    )
    doc.build(_content(_styles()))
    log.info("PDF generado en %s", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el CV PDF del candidato.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Ruta del PDF de salida (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_pdf(args.output)


if __name__ == "__main__":
    main()
