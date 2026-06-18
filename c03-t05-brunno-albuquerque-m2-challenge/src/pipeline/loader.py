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
from collections import defaultdict

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
#from langchain_text_loaders import TextLoader 
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

# ---------------------------------------------------------------------------
# Category labels derived from filename prefixes.
#
# NOTE: Order matters here — more specific prefixes must come BEFORE shorter
# ones that share the same start (e.g. 'product_catalog' before a hypothetical
# 'product' catch-all).  Python dicts preserve insertion order (3.7+), and
# _infer_category() returns on the first match.
# ---------------------------------------------------------------------------
_CATEGORY_MAP: dict[str, str] = {
    "product_manual":  "product_manual",   # product_manual_*.txt
    "product_catalog": "product_catalog",  # product_catalog.txt
    "service_catalog": "service_catalog",  # service_catalog.txt
    "support":         "support_article",  # support_*.txt
    "policy":          "policy",           # policy_*.txt
    "trade_in":        "policy",           # trade_in_program.txt
}


def _infer_category(filename: str) -> str:
    """Infer the document category from the filename prefix.

    Supported prefixes: see _CATEGORY_MAP above.
    Falls back to 'general' for unrecognised filenames.

    Args:
        filename: The bare filename (e.g. 'product_catalog.txt').

    Returns:
        One of the category values in _CATEGORY_MAP, or 'general'.
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
    - ``source``:     filename (e.g. 'product_catalog.txt')
    - ``category``:   one of the values in _CATEGORY_MAP, or 'general'
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
        1. Validate that *data_dir* exists; raise FileNotFoundError with a helpful message if not. (OK)
        2. Iterate over all *.txt files in the top-level directory (not subdirectories — tables/ is handled separately). (OK)
        3. Use ``TextLoader`` with ``encoding="utf-8"`` for .txt files. (OK)
        4. Use ``PyPDFLoader`` for .pdf files if any exist in your corpus. (OK)
         + 5. Attach metadata after loading (see keys above). (OK)
        6. Print a summary: document count and approximate total word count. (OK)
        7. Return the list of Documents. (OK)
        "To-do: implement load_documents() — see Stop 1 in m2-capstone-rag-knowledge-base.md"
    """
    # 1. Validate that data_dir exists
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: '{data_dir}'. "
            "Make sure you are running from the project root and the data/ folder exists."
        )

    documents: list[Document] = []

    # 2. Iterate over top-level .txt files only (tables/ subdirectory is skipped)
    txt_files = sorted(data_dir.glob("*.txt"))
    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not txt_files and not pdf_files:
        raise RuntimeError(
            f"No .txt or .pdf files found in '{data_dir}'. "
            "Check that the corpus files are present."
        )

    # 3. Load .txt files
    for filepath in txt_files:
        loader = TextLoader(str(filepath), encoding="utf-8")
        loaded = loader.load()  # returns list[Document] (usually 1 doc per file)

        filename = filepath.name
        category = _infer_category(filename)

        # 5. Attach metadata
        for doc in loaded:
            doc.metadata["source"]    = filename
            doc.metadata["category"]  = category
            doc.metadata["file_path"] = str(filepath)
            documents.append(doc)

    # 4. Load .pdf files (if any)
    for filepath in pdf_files:
        loader = PyPDFLoader(str(filepath))
        loaded = loader.load()

        filename = filepath.name
        category = _infer_category(filename)

        # 5. Attach metadata
        for doc in loaded:
            doc.metadata["source"]    = filename
            doc.metadata["category"]  = category
            doc.metadata["file_path"] = str(filepath)
            documents.append(doc)

    # 6. Print summary
    total_words = sum(len(doc.page_content.split()) for doc in documents)
    print(f"[loader] Loaded {len(documents)} document(s) from '{data_dir}'")
    print(f"[loader] Approximate total word count: {total_words:,}")

    # Show per-category breakdown
    from collections import Counter
    cats = Counter(doc.metadata["category"] for doc in documents)
    for cat, count in sorted(cats.items()):
        print(f"         {cat}: {count} file(s)")

    # 7. Return the list of Documents
    return documents


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
        docs:          Documents to split (output of :func:`load_documents`).
        chunk_size:    Maximum characters per chunk.  Defaults to :data:`CHUNK_SIZE`.
        chunk_overlap: Character overlap between consecutive chunks.
                       Defaults to :data:`CHUNK_OVERLAP`.

    Returns:
        A flat list of chunk :class:`~langchain_core.documents.Document` objects.
    Example::
        chunks = chunk_documents(docs)
        print(f"{len(chunks)} chunks, avg {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    TODO — Stop 1:
        1. Create a ``RecursiveCharacterTextSplitter`` with the given parameters. (OK)
        2. Call ``splitter.split_documents(docs)`` to get the raw chunks. (OK)
        3. Add ``chunk_index`` metadata: for each source file, number its chunks
           starting from 0.  Hint: group chunks by ``chunk.metadata['source']`` and enumerate. (OK)
        4. Print the summary. (OK)
        5. Return the augmented chunk list.
    """
    # 1. Create the splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # paragraph → line → word → char
    )

    # 2. Split all documents at once
    raw_chunks: list[Document] = splitter.split_documents(docs)

    # 3. Add chunk_index metadata — group chunks by their source file and
    #    assign a 0-based index within each source.
    source_counters: dict[str, int] = defaultdict(int)

    for chunk in raw_chunks:
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_index"] = source_counters[source]
        source_counters[source] += 1

    # 4. Print summary
    if raw_chunks:
        avg_len = sum(len(c.page_content) for c in raw_chunks) // len(raw_chunks)
    else:
        avg_len = 0

    print(f"\n[loader] Chunked {len(docs)} document(s) → {len(raw_chunks)} chunk(s)")
    print(f"[loader] chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"[loader] Average chunk length: {avg_len} characters")

    # Show per-source chunk count
    for source, count in sorted(source_counters.items()):
        print(f"         {source}: {count} chunk(s)")

    #5. Return the augmented chunk list
    return raw_chunks