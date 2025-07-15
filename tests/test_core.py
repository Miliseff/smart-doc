import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import chunk_text, build_prompt, retrieve, _build_index, embed_texts, _CHUNKS, _INDEX


def test_chunk_text_empty():
    assert chunk_text("", chunk_size=10, overlap=2) == []


def test_chunk_text_exact_and_overlap():
    text = "abcdefghij"
    assert chunk_text(text, chunk_size=5, overlap=2) == ["abcde", "defgh", "ghij", "ij"]


def test_chunk_text_guard_loop():
    text = "short"
    assert chunk_text(text, chunk_size=10, overlap=5) == ["short"]


@pytest.fixture(autouse=True)
def setup_index(monkeypatch):
    import main as m
    chunks = ["foo bar", "baz qux", "lorem ipsum"]
    idx = m._build_index(chunks)
    m._CHUNKS = chunks
    m._INDEX = idx
    m.EMBED_BACKEND = "local"
    yield


def test_retrieve_matches_first_chunk():
    results = retrieve("foo", top_k=1)
    assert len(results) == 1
    snippet, idx = results[0]
    assert idx == 0
    assert "foo bar" in snippet


def test_build_prompt_format():
    prompt = build_prompt("Q?", [("text passage", 2)])
    assert "(2) text passage" in prompt
    assert "Question: Q?" in prompt
