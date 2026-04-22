"""Tests del factory de retrievers (rama sin reranker)."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.rag.application.retrieval import build_retriever


def _fake_vs() -> FAISS:
    docs = [
        Document(page_content="Python y FastAPI", metadata={"source": "a.md"}),
        Document(page_content="LLMs y RAG con LangChain", metadata={"source": "b.md"}),
        Document(page_content="Docker y AWS", metadata={"source": "c.md"}),
    ]
    return FAISS.from_documents(docs, DeterministicFakeEmbedding(size=32))


def test_build_retriever_without_reranker_returns_top_k():
    vs = _fake_vs()
    retriever = build_retriever(vs, top_k=2, enable_reranker=False)

    docs = retriever.invoke("LLM")
    assert len(docs) == 2
    assert all("source" in d.metadata for d in docs)


def test_build_retriever_default_uses_settings():
    vs = _fake_vs()
    # Con el default (reranker desactivado) debe devolver un retriever plano
    # y no importar dependencias pesadas.
    retriever = build_retriever(vs, top_k=1)
    docs = retriever.invoke("Docker")
    assert len(docs) == 1
