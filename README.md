# smart-doc


A lightweight command‑line tool that converts any collection of text files into a searchable knowledge base and answers questions with clear, cited responses.

---


## Main features

* **Single‑command use** – `smartqa --input docs/guide.md --ask `.
* **Interactive mode** – Omit `--ask` to open a REPL for follow‑up questions.
* **Configurable embeddings** – Works with OpenAI models or any local `sentence‑transformers` model.
* **Fast retrieval** – Text is split into manageable chunks and indexed in FAISS by default .
* **Transparent answers** – Each answer is limited to \~200 words and includes inline citations so you can verify the source.
* **Cost and latency log** – Every request is logged to `qa_history.jsonl` with token usage and response time.

## How it works

```
load → chunk → embed → index
query → retrieve → build prompt → generate answer
```

## Installation

```bash
# Clone the repository
$ git clone https://github.com/youruser/smart-doc
$ cd smart-doc

# Install dependencies
$ poetry install  # or: pip install -r requirements.txt
```

> **Requirements:** Python 3.9 or later. If you plan to use hosted models set `OPENAI_API_KEY` in your environment.

## Quick start

```bash
# Index a document and open the interactive shell
$ smartqa --input docs/brief.txt

# One‑shot question
$ smartqa --input docs/brief.txt --ask 
```

Configuration values (chunk size, model name, top‑k, etc.) live in `config.yaml` and can be overridden from the command line.

## Testing

```bash
$ pytest
```

## Roadmap

* Streaming output option (`--stream`)
* Async batch mode
* `--score` flag to print similarity scores
* Docker image and minimal web UI

