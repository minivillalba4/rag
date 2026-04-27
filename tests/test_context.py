from src.context import build_context


def test_build_context_uses_normalized_sources():
    sources = [
        {"id": "0", "title": "profile", "content": "Primer chunk.", "metadata": {}},
        {"id": "1", "title": "profile", "content": "Segundo chunk.", "metadata": {}},
    ]

    assert build_context(sources) == "[0] Primer chunk.\n\n[1] Segundo chunk."
