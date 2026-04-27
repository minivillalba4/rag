from langchain_core.documents import Document

from src import pipeline


class FakePipelineVectorStore:
    def similarity_search(self, query: str, k: int = 3):
        return [
            Document(
                page_content="Ismael trabaja con Python, FastAPI y RAG.",
                metadata={"source": "profile.md", "title": "profile"},
            )
        ][:k]


def fake_answer_generator(context: str, question: str) -> str:
    return f"{question}\n{context}"


def test_rag_pipeline_returns_answer_and_sources():
    rag = pipeline.RagPipeline(
        vectorstore=FakePipelineVectorStore(),
        answer_generator=fake_answer_generator,
    )

    result = rag.answer_question("¿Qué tecnologías domina Ismael?")

    assert "answer" in result
    assert result["sources"] == [
        {
            "id": "0",
            "title": "profile",
            "content": "Ismael trabaja con Python, FastAPI y RAG.",
            "metadata": {"source": "profile.md", "title": "profile", "rank": 1},
        }
    ]


def test_answer_question_uses_cached_pipeline(monkeypatch):
    calls = {"build": 0}

    def fake_build_rag_pipeline(config=None):
        calls["build"] += 1
        return pipeline.ConversationalRagPipeline(vectorstore=FakePipelineVectorStore())

    pipeline.clear_pipeline_cache()
    monkeypatch.setattr(pipeline, "build_rag_pipeline", fake_build_rag_pipeline)
    monkeypatch.setattr(pipeline, "generate_answer", fake_answer_generator)

    first = pipeline.answer_question("Pregunta 1")
    second = pipeline.answer_question("Pregunta 2")

    assert first["sources"][0]["content"] == second["sources"][0]["content"]
    assert calls["build"] == 1

    pipeline.clear_pipeline_cache()
