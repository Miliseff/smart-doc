
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np 
import typer
from sentence_transformers import SentenceTransformer  
import faiss  
import os, openai
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = typer.Typer(add_completion=False, help="smart_doc – ask questions to your docs")

_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  
_INDEX: faiss.Index
_CHUNKS: List[str]


def _build_index(chunks: List[str]) -> faiss.Index:
    embeds = _MODEL.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    # FAISS cosine = inner product on normalised vectors
    faiss.normalize_L2(embeds)
    dim = embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeds)
    return index


def chunk_text(text: str, *, chunk_size: int = 500, overlap: int = 50) -> List[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    if text_len == 0:
        return chunks

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start = end - overlap if overlap else end
    return chunks


def retrieve(query: str, *, top_k: int = 5) -> List[Tuple[str, int]]:
    """Return *top_k* chunks most similar to *query*. Assumes global index."""
    if not _CHUNKS:
        return []
    q_vec = _MODEL.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    D, I = _INDEX.search(q_vec, top_k)
    return [( _CHUNKS[idx], int(idx)) for idx in I[0] if idx != -1]


def build_prompt(question: str, passages: List[Tuple[str, int]]) -> str:
    ctx = "\n".join(f"({i}) {p}" for p, i in passages)
    return (
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )


def call_llm(prompt: str, *, model: str = "gpt-3.5-turbo") -> str:
    resp = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def log_query(ts: float, question: str, *, tokens: int, cost: float, latency_ms: float, path: Path = Path("qa_history.jsonl")) -> None:
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
        "question": question,
        "tokens": tokens,
        "cost_usd": cost,
        "latency_ms": latency_ms,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")



@app.command()
def main(
    *,
    input: Path = typer.Option(..., "--input", exists=True, file_okay=True, readable=True, help="Input text file."),
    ask: Optional[str] = typer.Option(None, "--ask", help="Question to ask. If omitted, enters interactive mode."),
    top_k: int = typer.Option(5, help="How many passages to retrieve."),
    model: str = typer.Option("gpt-4o-mini", help="LLM model name (placeholder)."),
    chunk_size: int = typer.Option(500, help="Chunk size in characters."),
    overlap: int = typer.Option(50, help="Chunk overlap in characters."),
) -> None:  

    global _INDEX, _CHUNKS  
    text = input.read_text(encoding="utf-8", errors="replace")
    _CHUNKS = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    _INDEX = _build_index(_CHUNKS)

    def handle(q: str) -> None:
        t0 = time.perf_counter()
        passages = retrieve(q, top_k=top_k)
        prompt = build_prompt(q, passages)
        answer = call_llm(prompt, model=model)
        latency = (time.perf_counter() - t0) * 1000
        log_query(time.time(), q, tokens=len(prompt.split()), cost=0.0, latency_ms=latency)
        typer.echo(answer)
        typer.echo("\nSources:")
        for passage, idx in passages:
            snippet = passage.replace("\n", " ")[:80]
            typer.echo(f"({idx}) {snippet}…")

    if ask:
        handle(ask)
    else:
        typer.echo("Type 'exit' to quit.\n")
        while True:
            try:
                q = input("? ") 
            except (EOFError, KeyboardInterrupt):
                break
            if q.strip().lower() in {"exit", "quit", "q"}:
                break
            handle(q)


if __name__ == "__main__":
    app()