"""Smoke test del RAG chain sin depender de Ollama."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.rag.chain import build_rag_chain, format_history, format_sources


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


def test_condense_chain_rewrites_follow_up():
    rewritten = "¿En qué empresa trabajó con Python?"
    fake_llm = FakeListChatModel(responses=[rewritten])

    bundle = build_rag_chain(
        vectorstore=_fake_vectorstore(),
        llm=fake_llm,
        top_k=2,
    )

    history_text = format_history(
        [
            {"role": "user", "content": "¿Qué experiencia tiene con Python?"},
            {"role": "assistant", "content": "3 años en backend."},
        ]
    )
    result = bundle.condense_chain.invoke(
        {"chat_history": history_text, "question": "¿y en qué empresa?"}
    )
    assert result == rewritten


def test_format_history_truncates_and_skips_empty():
    history = [
        {"role": "user", "content": "p1"},
        {"role": "assistant", "content": "r1"},
        {"role": "user", "content": "p2"},
        {"role": "assistant", "content": "r2"},
        {"role": "user", "content": ""},
        {"role": "user", "content": "p3"},
    ]
    text = format_history(history, max_turns=2)

    assert "Usuario: p1" not in text
    assert "Usuario: p2" in text
    assert "Asistente: r2" in text
    assert "Usuario: p3" in text
    assert "Usuario: \n" not in text and not text.endswith("Usuario: ")


def test_format_sources_indexes_match_context_order():
    docs = [
        Document(
            page_content="Primer fragmento sobre LLMs.",
            metadata={"source": "cv.md", "page": 0},
        ),
        Document(
            page_content="Segundo fragmento sobre Docker.",
            metadata={"source": "proyectos.md"},
        ),
        Document(
            page_content="x" * 500,
            metadata={"source": "largo.md"},
        ),
    ]
    rendered = format_sources(docs)

    assert "**[1]** `cv.md` (p.1)" in rendered
    assert "**[2]** `proyectos.md`" in rendered
    assert "**[3]** `largo.md`" in rendered
    assert "> Primer fragmento sobre LLMs." in rendered
    assert "…" in rendered  # el tercer snippet se trunca
    assert rendered.index("[1]") < rendered.index("[2]") < rendered.index("[3]")
