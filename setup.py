from setuptools import setup

setup(
    name="smart-doc",
    version="0.1.0",
    py_modules=["main"],
    install_requires=[
    "torch>=2.1.0",
    "faiss-cpu>=1.7.4",
    "sentence-transformers>=2.6.0",
    "openai>=1.0.0",
    "typer>=0.16.0",
    "requests>=2.31.0",
    "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "smartqa=main:app",
        ],
    },
    author="",
    description="CLI tool to search and answer questions over text documents using embeddings and OpenAI",
    license="MIT",
)
