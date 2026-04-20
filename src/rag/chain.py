from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama

from .config import settings
from .prompts import prompt_template
from .vectorstore import load_or_build_vectorstore


@dataclass
class RagBundle:
    """Agrupa el chain de respuesta y el retriever subyacente.

    El retriever se expone aparte porque la UI necesita los documentos fuente
    para mostrarlos como citas, separado del flujo de generación.
    """

    chain: Runnable
    retriever: VectorStoreRetriever


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "desconocido")
        page = d.metadata.get("page")
        header = f"[{i}] {src}" + (f" (p.{page + 1})" if page is not None else "")
        parts.append(f"{header}\n{d.page_content.strip()}")
    return "\n\n".join(parts)


def build_rag_chain(
    vectorstore: FAISS | None = None,
    llm: BaseChatModel | None = None,
    top_k: int | None = None,
) -> RagBundle:
    vectorstore = vectorstore or load_or_build_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k or settings.top_k}
    )

    llm = llm or ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )

    chain: Runnable = (
        {
            "context": retriever | RunnableLambda(_format_context),
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return RagBundle(chain=chain, retriever=retriever)
