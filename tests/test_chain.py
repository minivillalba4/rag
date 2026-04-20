"""Smoke test del RAG chain sin depender de Ollama."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.rag.chain import build_rag_chain


def _fake_vectorstore() -> FAISS:
    docs = [
        Document(
            page_content=(
                "Juan ha trabajado 3 años en proyectos de LLMs y RAG, "
                "usando LangChain, FAISS y modelos de Ollama."
            ),
            metadata={"source": "cv.md", "page": 0},
        ),
        Document(
            page_content=(
                "Domina Docker, Python, FastAPI y tiene experiencia en AWS."
            ),
            metadata={"source": "cv.md", "page": 1},
        ),
    ]
    embeddings = DeterministicFakeEmbedding(size=64)
    return FAISS.from_documents(docs, embeddings)


def test_chain_runs_with_fake_components():
    expected = "Tiene 3 años de experiencia en LLMs."
    fake_llm = FakeListChatModel(responses=[expected])

    bundle = build_rag_chain(
        vectorstore=_fake_vectorstore(),
        llm=fake_llm,
        top_k=2,
    )

    result = bundle.chain.invoke("¿Cuánta experiencia tiene en LLMs?")
    assert result == expected


def test_retriever_returns_documents():
    bundle = build_rag_chain(
        vectorstore=_fake_vectorstore(),
        llm=FakeListChatModel(responses=["ok"]),
        top_k=2,
    )

    docs = bundle.retriever.invoke("LLMs")
    assert len(docs) == 2
    assert all("source" in d.metadata for d in docs)
