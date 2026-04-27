from langchain_core.documents import Document

from src.retriever import normalize_sources, retrieve_documents


class FakeVectorStore:
    def similarity_search(self, query: str, k: int = 3):
        return [
            Document(
                page_content=f"Resultado para {query}",
                metadata={"source": "profile.md", "title": "profile"},
            )
        ][:k]


def test_retrieve_documents_delegates_to_vectorstore():
    docs = retrieve_documents(FakeVectorStore(), "Python", k=1)

    assert len(docs) == 1
    assert docs[0].page_content == "Resultado para Python"


def test_normalize_sources_returns_api_friendly_shape():
    docs = [
        Document(
            page_content="Ismael trabaja con Python.",
            metadata={"source": "profile.md", "title": "profile"},
        )
    ]

    sources = normalize_sources(docs)

    assert sources == [
        {
            "id": "0",
            "title": "profile",
            "content": "Ismael trabaja con Python.",
            "metadata": {"source": "profile.md", "title": "profile", "rank": 1},
        }
    ]
