"""Test del dataset dorado de evaluación.

No ejecuta RAGAS (dependencia pesada opcional): sólo valida que el JSONL
parsee y tenga los campos requeridos.
"""

import json
from pathlib import Path


GOLDEN = Path(__file__).parent / "eval" / "golden_dataset.jsonl"


def _load_rows() -> list[dict]:
    rows = []
    with GOLDEN.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append((i, json.loads(line)))
    return rows


def test_golden_dataset_exists():
    assert GOLDEN.exists(), f"Falta {GOLDEN}"


def test_golden_rows_have_required_fields():
    rows = _load_rows()
    assert rows, "El dataset está vacío"
    for i, row in rows:
        assert isinstance(row.get("question"), str) and row["question"].strip(), (
            f"Línea {i}: 'question' ausente o vacía"
        )
        assert isinstance(row.get("ground_truth"), str) and row["ground_truth"].strip(), (
            f"Línea {i}: 'ground_truth' ausente o vacía"
        )


def test_golden_covers_guardrail_case():
    """Al menos una fila debe testear la respuesta de 'no sé' del RAG."""
    rows = _load_rows()
    assert any(
        "No tengo esa información" in row["ground_truth"] for _, row in rows
    ), "Falta al menos una pregunta fuera de contexto para validar el guardrail"
