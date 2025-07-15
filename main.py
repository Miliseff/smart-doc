from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import requests
import typer
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

app = typer.Typer(add_completion=False)

EMBED_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_API_TOKEN")

if EMBED_BACKEND == "openai":
    _embed_client = OpenAI(api_key=OPENAI_KEY)
elif EMBED_BACKEND == "hf":
    _hf_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBED_MODEL}"
else:
    _st_model = SentenceTransformer(EMBED_MODEL)

_llm_client = OpenAI(api_key=OPENAI_KEY)
PRICING = {"gpt-3.5-turbo": 0.002, "gpt-4o-mini": 0.003}
_INDEX: faiss.Index = None
_CHUNKS: List[str] = []


def embed_texts(texts: List[str]) -> np.ndarray:
    if EMBED_BACKEND == "openai":
        resp = _embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
        embeds = [d.embedding for d in resp.data]
        arr = np.array(embeds, dtype=np.float32)
    elif EMBED_BACKEND == "hf":
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        arr = np.array(requests.post(_hf_url, json=texts, headers=headers).json(), dtype=np.float32)
    else:
        arr = _st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return arr


def _build_index(chunks: List[str]) -> faiss.Index:
    embeds = embed_texts(chunks)
    faiss.normalize_L2(embeds)
    dim = embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeds)
    return index


def chunk_text(text: str, *, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if chunk_size <= 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size or overlap configuration")
    out: List[str] = []
    start = 0
    length = len(text)
    if length == 0:
        return out
    while start < length:
        end = min(start + chunk_size, length)
        out.append(text[start:end])
        next_start = end - overlap
        if next_start <= start:
            break
        start = next_start
    return out


def retrieve(query: str, *, top_k: int = 5) -> List[Tuple[str, int]]:
    if not _CHUNKS:
        return []
    q_emb = embed_texts([query])
    faiss.normalize_L2(q_emb)
    D, I = _INDEX.search(q_emb, top_k)
    return [(_CHUNKS[i], int(i)) for i in I[0] if i != -1]


def build_prompt(q: str, passages: List[Tuple[str, int]]) -> str:
    ctx = "\n".join(f"({i}) {p}" for p, i in passages)
    return f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"


def call_llm(prompt: str, *, model: str = "gpt-3.5-turbo") -> Tuple[str, object]:
    resp = _llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip(), resp.usage


def log_query(ts: float, question: str, *, tokens: int, cost_usd: float, latency_ms: float, path: Path = Path("qa_history.jsonl")) -> None:
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
        "question": question,
        "tokens": tokens,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.command()
def main(
    *,
    input: Path = typer.Option(..., "--input", exists=True, help="Input text file."),
    ask: Optional[str] = typer.Option(None, "--ask", help="Question to ask. If omitted, interactive mode."),
    top_k: int = typer.Option(5, "--top-k", help="How many passages to retrieve."),
    model: str = typer.Option("gpt-3.5-turbo", "--model", help="LLM model name."),
    chunk_size: int = typer.Option(500, "--chunk-size", help="Chunk size in characters."),
    overlap: int = typer.Option(50, "--overlap", help="Chunk overlap in characters."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output."),
) -> None:
    global _INDEX, _CHUNKS
    if debug:
        typer.echo(f"Using embed backend: {EMBED_BACKEND}, model: {EMBED_MODEL}", err=True)
    text = input.read_text(encoding="utf-8", errors="replace")
    _CHUNKS = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    idx_path = input.with_suffix(".idx")
    if idx_path.exists():
        _INDEX = faiss.read_index(str(idx_path))
    else:
        _INDEX = _build_index(_CHUNKS)
        faiss.write_index(_INDEX, str(idx_path))

    def handle(q: str) -> None:
        start = time.perf_counter()
        passages = retrieve(q, top_k=top_k)
        prompt = build_prompt(q, passages)
        answer, usage = call_llm(prompt, model=model)
        total_tokens = getattr(usage, 'total_tokens', 0)
        cost_usd = total_tokens * (PRICING.get(model, 0.0) / 1000)
        latency = (time.perf_counter() - start) * 1000
        log_query(time.time(), q, tokens=total_tokens, cost_usd=cost_usd, latency_ms=latency)
        typer.echo(answer)
        typer.echo("\nSources:")
        for p, i in passages:
            snippet = p.replace("\n", " ")[:80]
            typer.echo(f"({i}) {snippet}…")

    if ask:
        handle(ask)
    else:
        typer.echo("Type 'exit' to quit.")
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
