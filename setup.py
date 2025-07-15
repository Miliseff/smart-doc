from setuptools import setup

setup(
    name="smart-doc",
    version="0.1.0",
    py_modules=["main"],
    install_requires=[
        "typer",
        "openai>=1.0.0",
        "sentence-transformers",
        "faiss-cpu",
        "numpy",
        "requests",
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
