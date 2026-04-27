from langchain_core.documents import Document

from src.chunking import split_documents


def test_split_documents_preserves_metadata():
    document = Document(
        page_content="Python FastAPI RAG " * 20,
        metadata={"source": "profile.md", "title": "profile"},
    )

    chunks = split_documents([document], chunk_size=40, chunk_overlap=5)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "profile.md" for chunk in chunks)
    assert all(chunk.metadata["title"] == "profile" for chunk in chunks)
