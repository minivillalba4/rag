from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .loaders import load_documents


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embed_model,
        base_url=settings.ollama_base_url,
    )


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_index(data_dir: Path | None = None, index_dir: Path | None = None) -> FAISS:
    """Carga documentos, los trocea, embebe y persiste el índice FAISS."""
    data_dir = data_dir or settings.data_dir
    index_dir = index_dir or settings.index_dir

    docs = load_documents(data_dir)
    chunks = split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, _get_embeddings())

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    return vectorstore


def add_documents_to_index(
    new_docs: list[Document],
    index_dir: Path | None = None,
) -> FAISS:
    """Añade documentos a un índice existente (o lo crea) y lo persiste."""
    from .vectorstore import load_or_build_vectorstore

    index_dir = index_dir or settings.index_dir
    vectorstore = load_or_build_vectorstore()

    chunks = split_documents(new_docs)
    vectorstore.add_documents(chunks)
    vectorstore.save_local(str(index_dir))
    return vectorstore
