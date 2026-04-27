from pathlib import Path

from src.ingestion import load_markdown_document


def test_load_markdown_document_reads_content_and_metadata():
    profile_path = "data/profile/profile.md"
    document = load_markdown_document(profile_path)

    assert "Ismael" in document.page_content
    assert Path(document.metadata["source"]) == Path(profile_path)
    assert document.metadata["title"] == "profile"
    assert document.metadata["type"] == "markdown"
