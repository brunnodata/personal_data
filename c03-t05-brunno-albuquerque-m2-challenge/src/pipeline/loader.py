"""
pipeline/loader.py — Document loading and chunking for the TechStore Plus corpus.

Position in the architecture:
    data/  →  loader.py  →  vectorstore.py  →  retriever  →  LLM

Stop 1 (W4): Implement load_documents() and chunk_documents().

Design decisions you should document in your docstrings:
- Why RecursiveCharacterTextSplitter? It respects paragraph and sentence
  boundaries (trying '\\n\\n', '\\n', ' ' in order), which keeps semantically
  coherent chunks. A fixed-size splitter ignores structure entirely.
- Why add metadata (source, category, chunk_index)? The retriever returns
  Document objects; metadata lets the writer cite sources and lets the
  guardrail verifier trace claims back to specific files.
- Why chunk_size=500 as the default? See docs/chunk-experiment.md — you will
  empirically validate this in Stop 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Constants — change these values to experiment; do not use magic numbers
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 500
"""Default chunk size in characters for RecursiveCharacterTextSplitter."""

CHUNK_OVERLAP: int = 50
"""Default overlap between adjacent chunks.

A 10% overlap (50 chars on a 500-char chunk) reduces the risk of splitting
a critical sentence across two chunks while keeping index size reasonable.
"""

DATA_DIR: Path = Path("data")
"""Default path to the TechStore Plus corpus directory, relative to project root."""

# Category labels derived from filename prefixes
_CATEGORY_MAP: dict[str, str] = {
    "product_manual": "product_manual",
    "support": "support_article",
    "policy": "policy",
}


def _infer_category(filename: str) -> str:
    """Infer the document category from the filename prefix.

    Supported prefixes: 'product_manual_*', 'support_*', 'policy_*'.
    Falls back to 'general' for unrecognised filenames.

    Args:
        filename: The bare filename (e.g. 'policy_return_policy.txt').

    Returns:
        One of 'product_manual', 'support_article', 'policy', or 'general'.
    """
    for prefix, category in _CATEGORY_MAP.items():
        if filename.startswith(prefix):
            return category
    return "general"


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load all .txt and .pdf files from *data_dir* and attach metadata.

    Only the top-level directory is loaded here; CSV tables are handled by
    ``src.multimodal.table_retriever.TableRetriever`` (Stop 3).

    Each returned Document has the following metadata keys:
    - ``source``:      filename (e.g. 'policy_return_policy.txt')
    - ``category``:   one of 'product_manual', 'support_article', 'policy', 'general'
    - ``file_path``:  full path as string

    After loading, print a summary:
    - Number of documents loaded
    - Total word count (approximate)

    Args:
        data_dir: Path to the corpus directory.  Defaults to :data:`DATA_DIR`.

    Returns:
        A list of :class:`~langchain_core.documents.Document` objects, one per file.

    Raises:
        FileNotFoundError: If *data_dir* does not exist.
        RuntimeError: If no supported files are found in *data_dir*.

    Example::

        docs = load_documents()
        print(f"Loaded {len(docs)} documents")

    TODO — Stop 1:
        1. Validate that *data_dir* exists; raise FileNotFoundError with a
           helpful message if not.
        2. Iterate over all *.txt files in the top-level directory (not
           subdirectories — tables/ is handled separately).
        3. Use ``TextLoader`` with ``encoding="utf-8"`` for .txt files.
        4. Use ``PyPDFLoader`` for .pdf files if any exist in your corpus.
        5. Attach metadata after loading (see keys above).
        6. Print a summary: document count and approximate total word count.
        7. Return the list of Documents.
    """
    raise NotImplementedError(
        "TODO: implement load_documents() — see Stop 1 in m2-capstone-rag-knowledge-base.md"
    )


def chunk_documents(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split *docs* into overlapping chunks and add ``chunk_index`` metadata.

    Uses :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter`,
    which splits on paragraph breaks ('\\n\\n') first, then line breaks ('\\n'),
    then spaces, then characters — preserving semantic boundaries as long as
    possible.

    Each returned chunk inherits the parent document's metadata and gains:
    - ``chunk_index``: integer position of this chunk within its source document
      (0-indexed).

    After chunking, print a summary:
    - Total chunk count
    - Average chunk length in characters

    Args:
        docs:           Documents to split (output of :func:`load_documents`).
        chunk_size:     Maximum characters per chunk.  Defaults to :data:`CHUNK_SIZE`.
        chunk_overlap:  Character overlap between consecutive chunks.
                        Defaults to :data:`CHUNK_OVERLAP`.

    Returns:
        A flat list of chunk :class:`~langchain_core.documents.Document` objects.

    Example::

        chunks = chunk_documents(docs)
        print(f"{len(chunks)} chunks, avg {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    TODO — Stop 1:
        1. Create a ``RecursiveCharacterTextSplitter`` with the given parameters.
        2. Call ``splitter.split_documents(docs)`` to get the raw chunks.
        3. Add ``chunk_index`` metadata: for each source file, number its chunks
           starting from 0.  Hint: group chunks by ``chunk.metadata['source']``
           and enumerate.
        4. Print the summary.
        5. Return the augmented chunk list.
    """
    raise NotImplementedError(
        "TODO: implement chunk_documents() — see Stop 1 in m2-capstone-rag-knowledge-base.md"
    )
