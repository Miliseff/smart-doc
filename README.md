# smart-doc

A lightweight CLI that turns text files into a searchable knowledge base and answers questions with inline citations.

---

## Setup

```bash
# Clone  repository
$ git clone https://github.com/Miliseff/smart-doc
$ cd smart-doc

# Install dependencies (choose one)
$ poetry install           
# or
$ pip install -r requirements.txt
```

> **Prerequisite:** Python 3.9 + and a C compiler (FAISS).

---

## Environment Variables

| Variable            | Purpose                                   | Default |
| ------------------- | ----------------------------------------- | ------- |
| `EMBEDDING_BACKEND` | `local`, `openai`, or `hf` for embeddings | `local` |
| `EMBEDDING_MODEL`   | Model name (e.g. `all-MiniLM-L6-v2`)      |         |
| `OPENAI_API_KEY`    | Required when `EMBEDDING_BACKEND=openai`  |         |
| `HF_API_TOKEN`      | Required when `EMBEDDING_BACKEND=hf`      |         |

Set variables in your shell or a `.env` file:

```bash
export EMBEDDING_BACKEND=openai
export OPENAI_API_KEY=sk-…
```

---

## Example Commands

```bash
# 1 Ask a one question
smartqa --input docs/spec.txt --ask "What is the main goal?"

# 2Open the interactive shell
smartqa --input docs/spec.txt

# 3 Index an entire directory 
smartqa --input "./handbook/**/*.md" --chunk-size 400

# 4 Use a local embedding model
export EMBEDDING_BACKEND=local
export EMBEDDING_MODEL=intfloat/e5-small-v2
smartqa --input README.md --ask "List the core features."

# 5 Stream answer tokens live and print similarity scores
smartqa --input docs/design.pdf --ask "Name the design principles" --stream --score
```

## Uso con Docker

### Construir la imagen

```bash
# Desde la raíz del proyecto
docker build -t smart-doc:latest .
```

### Ejecutar en REPL

```bash
docker run --rm -it \
  -v "$(pwd)/docs:/data" \
  smart-doc:latest --input /data/demo.txt
```

### Pregunta one‑shot

```bash
docker run --rm \
  -v "$(pwd)/docs:/data" \
  smart-doc:latest --input /data/demo.txt --ask "¿Qué problema resuelve?"
```

### Imagen publicada

```bash
# Descarga la imagen desde Docker Hub
docker pull miliseff/smart-doc:0.1
```

[Ver miliseff/smart-doc en Docker Hub](https://hub.docker.com/r/miliseff/smart-doc)
