from __future__ import annotations
import json, os, sys, time
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import typer
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = typer.Typer(add_completion=False)

_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
_INDEX: faiss.Index
_CHUNKS: List[str]

def _build_index(chunks: List[str]) -> faiss.Index:
    typer.echo("dbg: building index", err=True)
    s = time.perf_counter()
    embeds = _MODEL.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(embeds)
    dim = embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeds)
    typer.echo(f"dbg: index built {time.perf_counter()-s:.2f}s", err=True)
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
        next_start = end - overlap if overlap else end
        if next_start <= start:
            break
        start = next_start

    return chunks


def retrieve(query: str, *, top_k: int = 5) -> List[Tuple[str, int]]:
    if not _CHUNKS:
        return []
    q_vec = _MODEL.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    D, I = _INDEX.search(q_vec, top_k)
    return [(_CHUNKS[i], int(i)) for i in I[0] if i != -1]

def build_prompt(q: str, passages: List[Tuple[str, int]]) -> str:
    ctx = "\n".join(f"({i}) {p}" for p, i in passages)
    return f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"

def call_llm(prompt: str, *, model: str = "gpt-3.5-turbo") -> str:
    typer.echo("dbg: LLM call", err=True)
    s = time.perf_counter()
    rsp = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.2,
    )
    typer.echo(f"dbg: LLM done {time.perf_counter()-s:.2f}s", err=True)
    return rsp.choices[0].message.content.strip()

def log_query(ts: float, q: str, *, tokens: int, latency: float, path: Path = Path("qa_history.jsonl")) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": ts, "q": q, "tok": tokens, "lat_ms": latency}) + "\n")

@app.command()
def main(
    *,
    input: Path = typer.Option(..., "--input", exists=True),
    ask: Optional[str] = typer.Option(None, "--ask"),
    top_k: int = typer.Option(5),
    model: str = typer.Option("gpt-3.5-turbo"),
    chunk_size: int = typer.Option(500),
    overlap: int = typer.Option(50),
) -> None:
    global _INDEX, _CHUNKS
    typer.echo("dbg: reading file", err=True)
    text = input.read_text(encoding="utf-8", errors="replace")
    _CHUNKS = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    _INDEX = _build_index(_CHUNKS)

    def handle(q: str) -> None:
        t0 = time.perf_counter()
        typer.echo("dbg: retrieve", err=True)
        passages = retrieve(q, top_k=top_k)
        prompt = build_prompt(q, passages)
        answer = call_llm(prompt, model=model)
        lat = (time.perf_counter() - t0) * 1000
        log_query(time.time(), q, tokens=len(prompt.split()), latency=lat)
        typer.echo(answer)
        typer.echo("\nSources:")
        for p, i in passages:
            snippet = p.replace("\n", " ")
            typer.echo(f"({i}) {snippet[:80]}…")

    if ask:
        handle(ask)
    else:
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
