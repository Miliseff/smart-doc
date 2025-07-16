# smart-doc

## Installation

```bash
# Clone the repository
$ git clone https://github.com/Miliseff/smart-doc
$ cd smart-doc

# Install dependencies
$ poetry install      # or: pip install -r requirements.txt
```

> **Requirements:** Python 3.9 + and `gcc`/`clang` for compiling FAISS. Set `OPENAI_API_KEY` in your environment if you plan to use hosted models.

## Testing

```bash
$ pytest
```

## Environment Variables

| Variable            | Purpose                                                   | Default            |
| ------------------- | --------------------------------------------------------- | ------------------ |
| `EMBEDDING_BACKEND` | Select the embedding service: `local`, `openai`, or `hf`. | `local`            |
| `EMBEDDING_MODEL`   | Name of the embedding model to load or call.              | `all-MiniLM-L6-v2` |
| `OPENAI_API_KEY`    | Required when using OpenAI embeddings or chat models.     | —                  |
| `HF_API_TOKEN`      | Required when `EMBEDDING_BACKEND=hf`.                     | —                  |

Set variables in your shell or a `.env` file:

```bash
export EMBEDDING_BACKEND=openai
export OPENAI_API_KEY=sk-…
```

## Example Commands

```bash
# 1. Ask a one  question
smartqa --input docs/spec.txt --ask "What is the main goal?"

# 2.  Open the interactive shell
smartqa --input docs/spec.txt

# 3. Index an entire directory 
smartqa --input "./handbook/**/*.md" --chunk-size 400

# 4. Use a local embedding model
export EMBEDDING_BACKEND=local
export EMBEDDING_MODEL=intfloat/e5-small-v2
smartqa --input README.md --ask "List the core features."

# 5. Stream answer tokens live 
smartqa --input docs/design.pdf --ask "Name the design principles" --stream --score
```
