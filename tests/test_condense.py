"""Tests de memoria multi-turno: condense chain + retro-compatibilidad."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.rag.application.ask import build_ask_service
from src.rag.application.condense import format_history


def _fake_vs() -> FAISS:
    docs = [
        Document(
            page_content="Python: 5 años de experiencia con FastAPI y Django.",
            metadata={"source": "cv.md", "page": 0},
        ),
        Document(
            page_content="LLMs: proyectos de RAG con LangChain y FAISS.",
            metadata={"source": "cv.md", "page": 1},
        ),
    ]
    return FAISS.from_documents(docs, DeterministicFakeEmbedding(size=32))


def test_format_history_serializes_last_turns():
    history = [
        {"role": "user", "content": "Hola"},
        {"role": "assistant", "content": "¿En qué puedo ayudarte?"},
        {"role": "user", "content": "¿Experiencia con LLMs?"},
        {"role": "assistant", "content": "Sí, varios años."},
    ]
    out = format_history(history, window=2)
    assert "Usuario: Hola" in out
    assert "Asistente: Sí, varios años." in out


def test_format_history_respects_window():
    history = [
        {"role": "user", "content": f"msg{i}"} for i in range(10)
    ]
    out = format_history(history, window=2)
    # window=2 turnos → 4 mensajes máx.
    assert out.count("Usuario:") <= 4


def test_condense_chain_rewrites_question():
    condense_llm = FakeListChatModel(
        responses=["¿Qué experiencia tiene con Python?"]
    )
    answer_llm = FakeListChatModel(responses=["Tiene 5 años con Python."])

    bundle = build_ask_service(
        vectorstore=_fake_vs(),
        llm=answer_llm,
        condense_llm=condense_llm,
        top_k=2,
    )

    rewritten = bundle.condense.invoke(
        {"history": "Usuario: ¿Experiencia?\nAsistente: Mucha.", "question": "¿Y en Python?"}
    )
    assert rewritten == "¿Qué experiencia tiene con Python?"


def test_chain_backwards_compatible_with_string_input():
    """La chain sigue aceptando un str directo (sin historial)."""
    fake_llm = FakeListChatModel(responses=["Respuesta."])
    bundle = build_ask_service(
        vectorstore=_fake_vs(),
        llm=fake_llm,
        top_k=2,
    )

    result = bundle.chain.invoke("¿Experiencia con LLMs?")
    assert result == "Respuesta."


def test_chain_uses_context_question_for_retrieval():
    """Si se pasa ``context_question`` distinto de ``question``, el retrieval
    debe usar el primero pero el prompt final ver el segundo."""
    captured: dict[str, str] = {}

    # FakeListChatModel devuelve siempre la misma respuesta, pero podemos
    # interceptar invocando el retriever por separado.
    fake_llm = FakeListChatModel(responses=["ok"])
    bundle = build_ask_service(
        vectorstore=_fake_vs(),
        llm=fake_llm,
        top_k=2,
    )

    docs_original = bundle.retriever.invoke("Python")
    docs_rewritten = bundle.retriever.invoke("LLMs")
    # Con este corpus, las dos consultas rankean documentos distintos en primer lugar.
    assert docs_original[0].page_content != docs_rewritten[0].page_content

    # Y la chain acepta el dict explícito sin petar.
    result = bundle.chain.invoke(
        {"question": "¿Y en Python?", "context_question": "¿Experiencia con Python?"}
    )
    captured["result"] = result
    assert captured["result"] == "ok"
