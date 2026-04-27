from src import generation
from src.generation import build_user_input, generate_answer


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


def test_generate_answer_returns_fallback_without_context():
    answer = generate_answer("", "¿Qué sabe hacer Ismael?")

    assert "No tengo información suficiente" in answer


def test_build_user_input_includes_question_and_context():
    prompt = build_user_input("Ismael trabaja con RAG.", "¿Qué sabe hacer Ismael?")

    assert "¿Qué sabe hacer Ismael?" in prompt
    assert "Ismael trabaja con RAG." in prompt


def test_generate_answer_calls_inference_client(monkeypatch):
    captured = {}

    def fake_chat_completion(messages, model=None, **kwargs):
        captured["messages"] = messages
        captured["model"] = model
        return FakeResponse("Respuesta generada.")

    monkeypatch.setattr(generation, "chat_completion_with_fallback", fake_chat_completion)

    answer = generate_answer(
        "Ismael trabaja con RAG.",
        "¿Qué sabe hacer Ismael?",
        model_name="test-model",
    )

    assert "Respuesta generada." in answer
    assert captured["model"] == "test-model"
    assert any("Ismael trabaja con RAG." in m["content"] for m in captured["messages"])
