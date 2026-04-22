"""Evalúa el RAG sobre un dataset dorado usando RAGAS.

Uso:
    # Smoke test con 3 preguntas
    python -m scripts.run_eval --sample 3

    # Eval completa (usa top_k / ENABLE_RERANKER del entorno)
    python -m scripts.run_eval

    # Comparar configuraciones
    ENABLE_RERANKER=false python -m scripts.run_eval --tag baseline
    ENABLE_RERANKER=true  python -m scripts.run_eval --tag reranker

Por defecto el modelo juez es el mismo `ollama_model`. Para mejor calidad
de juicio, usa un modelo mayor: ``EVAL_JUDGE_MODEL=llama3.1:70b``.

Requiere (opcional): ``pip install ragas==0.2.10 datasets==3.1.0``
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.rag.application.ask import build_ask_service
from src.rag.config import settings
from src.rag.logging_config import configure_logging


logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "tests" / "eval" / "golden_dataset.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "eval" / "results"


def load_dataset(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "question" not in row or "ground_truth" not in row:
                raise ValueError(
                    f"{path}:{i} — fila sin 'question' o 'ground_truth': {row}"
                )
            rows.append(row)
    return rows


def generate_predictions(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Corre el RAG sobre cada pregunta y recolecta answer + contexts."""
    bundle = build_ask_service()
    predictions: list[dict[str, Any]] = []

    for i, row in enumerate(rows, start=1):
        q = row["question"]
        t0 = time.perf_counter()
        docs = bundle.retriever.invoke(q)
        answer = bundle.chain.invoke(q)
        dt = time.perf_counter() - t0
        logger.info(
            "[%d/%d] %.1fs · %d docs · %s",
            i,
            len(rows),
            dt,
            len(docs),
            q[:60],
        )
        predictions.append(
            {
                "question": q,
                "answer": answer,
                "contexts": [d.page_content for d in docs],
                "ground_truth": row["ground_truth"],
            }
        )
    return predictions


def run_ragas(predictions: list[dict[str, Any]], judge_model: str) -> Any:
    """Evalúa con RAGAS. Importa perezosamente para no exigir la dependencia."""
    try:
        from datasets import Dataset
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )
    except ImportError as e:
        logger.error(
            "Faltan dependencias de evaluación. Instala con: "
            "pip install ragas==0.2.10 datasets==3.1.0"
        )
        raise SystemExit(1) from e

    ds = Dataset.from_list(
        [
            {
                "user_input": p["question"],
                "response": p["answer"],
                "retrieved_contexts": p["contexts"],
                "reference": p["ground_truth"],
            }
            for p in predictions
        ]
    )

    judge = LangchainLLMWrapper(
        ChatOllama(
            model=judge_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
        )
    )
    judge_emb = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.ollama_base_url,
        )
    )

    metrics = [
        Faithfulness(llm=judge),
        AnswerRelevancy(llm=judge, embeddings=judge_emb),
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
    ]

    logger.info("Lanzando RAGAS con juez=%s (%d preguntas)", judge_model, len(predictions))
    return evaluate(dataset=ds, metrics=metrics, llm=judge, embeddings=judge_emb)


def write_csv(
    predictions: list[dict[str, Any]],
    scores: list[dict[str, float]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted({k for row in scores for k in row.keys()})
    fieldnames = ["question", "answer", "ground_truth", "n_contexts", *metric_keys]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pred, score in zip(predictions, scores):
            w.writerow(
                {
                    "question": pred["question"],
                    "answer": pred["answer"],
                    "ground_truth": pred["ground_truth"],
                    "n_contexts": len(pred["contexts"]),
                    **{k: score.get(k, "") for k in metric_keys},
                }
            )


def print_summary(result: Any, predictions: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print(f"EVALUACIÓN RAGAS · {len(predictions)} preguntas")
    print(f"Modelo respuesta: {settings.ollama_model}")
    print(f"Embeddings:       {settings.embed_model}")
    print(f"top_k={settings.top_k} · reranker={'on' if settings.enable_reranker else 'off'}")
    print("-" * 60)
    # RAGAS >=0.2 devuelve un objeto con `.scores` (lista de dicts por muestra).
    per_sample = list(result.scores) if hasattr(result, "scores") else []
    if per_sample:
        keys = sorted({k for row in per_sample for k in row.keys()})
        for k in keys:
            vals = [row[k] for row in per_sample if row.get(k) is not None]
            if vals:
                avg = sum(vals) / len(vals)
                print(f"  {k:35s} {avg:.3f}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el RAG con RAGAS.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample", type=int, default=None, help="Limitar a N preguntas.")
    parser.add_argument(
        "--judge-model",
        default=settings.eval_judge_model or settings.ollama_model,
        help="Modelo Ollama usado como juez (default: OLLAMA_MODEL).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default=None, help="Sufijo para el CSV resultante.")
    args = parser.parse_args()

    configure_logging()

    rows = load_dataset(args.dataset)
    if args.sample is not None:
        rows = rows[: args.sample]
    logger.info("Dataset cargado: %d preguntas desde %s", len(rows), args.dataset)

    predictions = generate_predictions(rows)
    result = run_ragas(predictions, judge_model=args.judge_model)

    per_sample = list(result.scores) if hasattr(result, "scores") else []
    print_summary(result, predictions)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_path = args.output_dir / f"eval_{timestamp}{tag}.csv"
    write_csv(predictions, per_sample, out_path)
    logger.info("Resultados por muestra guardados en %s", out_path)


if __name__ == "__main__":
    sys.exit(main())
