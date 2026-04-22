"""Adapter: ChatOllama → domain.LLM."""

from __future__ import annotations

from typing import AsyncIterator, Iterator, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from ..config import settings
from ..domain.entities import ChatMessage
from ..domain.ports import LLM


def build_chat_llm(temperature: float = 0.2) -> ChatOllama:
    """Fabrica un ChatOllama leyendo modelo y URL de settings."""
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )


_ROLE_TO_MESSAGE = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
}


def _to_langchain(messages: Sequence[ChatMessage]) -> list[BaseMessage]:
    return [_ROLE_TO_MESSAGE[m.role](content=m.content) for m in messages]


class OllamaChatLLM(LLM):
    """Adapter que implementa `LLM` sobre un `BaseChatModel` de LangChain."""

    def __init__(self, model: BaseChatModel | None = None) -> None:
        self._model = model or build_chat_llm()

    def invoke(self, messages: Sequence[ChatMessage]) -> str:
        return self._model.invoke(_to_langchain(messages)).content

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        for chunk in self._model.stream(_to_langchain(messages)):
            yield _chunk_text(chunk)

    async def ainvoke(self, messages: Sequence[ChatMessage]) -> str:
        msg = await self._model.ainvoke(_to_langchain(messages))
        return msg.content

    async def astream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        async for chunk in self._model.astream(_to_langchain(messages)):
            yield _chunk_text(chunk)


def _chunk_text(chunk: AIMessageChunk | BaseMessage) -> str:
    content = getattr(chunk, "content", chunk)
    return content if isinstance(content, str) else str(content)
