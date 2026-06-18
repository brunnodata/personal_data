"""
multimodal/table_retriever.py — Table-grounded retrieval from CSV files.

Position in the architecture:
    data/tables/*.csv  →  TableRetriever  →  TechStoreRAGAgent (merged evidence)

Stop 3 (W6): Implement TableRetriever.

WHY TABLE RETRIEVAL (for your docstring)?
    Numeric and comparison queries ("Which laptop has the most storage?",
    "How much does the Laptop Pro X1 cost?") require exact cell values.
    A prose embedding retriever may rank a paragraph that mentions 512 GB
    in context lower than a paragraph about storage best practices.

    TableRetriever converts each CSV row into a LangChain Document:
    - page_content: a natural-language serialisation of the row
      (e.g. "model=Laptop Pro X1, ram_gb=16, storage_gb=512, price_usd=1299,
              warranty_tier=premium")
    - metadata: row_index, column_names, source_file, table_citation

    Retrieval uses simple keyword/column matching or semantic similarity over
    the serialised rows, then returns Documents with table citations
    ([TB:filename:rowN]) so the writer prompt can cite them correctly.

    Numeric comparison (e.g. "most storage") is handled by sorting the parsed
    column values rather than relying on embedding similarity alone.

CITATION FORMAT:
    [TB:laptop_specs.csv:row2] — means row index 2 of laptop_specs.csv
    Include this citation in the Document's metadata as ``table_citation``.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TABLES_DIR: Path = Path("data/tables")
"""Default directory containing CSV table files."""

TABLE_RETRIEVAL_K: int = 3
"""Default number of table rows to return per query."""


class TableRetriever:
    """Parse CSV tables and retrieve relevant rows as Documents.

    Tables are loaded once via :meth:`load_tables` and cached in memory as
    lists of row Documents.  :meth:`retrieve` then finds the most relevant
    rows for a given query.

    Attributes:
        _tables: Dict mapping filename → list of row Documents.

    Example::

        tr = TableRetriever()
        tr.load_tables(Path("data/tables"))
        docs = tr.retrieve("How much RAM does the Laptop Pro X1 have?")
        for doc in docs:
            print(doc.page_content, doc.metadata["table_citation"])

    Note:
        This implementation covers Option A (table grounding) from the capstone
        brief.  Option B (image captions) would follow the same interface but
        use an ``ImageRetriever`` class instead.
    """

    def __init__(self) -> None:
        self._tables: dict[str, list[Document]] = {}

    def load_tables(self, tables_dir: Path = TABLES_DIR) -> None:
        """Parse all CSV files in *tables_dir* into row-level Documents.

        For each CSV file:
        - Read all rows using the ``csv`` standard library (no pandas required,
          but pandas is fine if you prefer).
        - Serialise each row as a natural-language string:
              "model=Laptop Pro X1, ram_gb=16, storage_gb=512, price_usd=1299,
               warranty_tier=premium"
        - Create a Document with:
              page_content = the serialised row string
              metadata = {
                  "source":          filename (e.g. "laptop_specs.csv"),
                  "row_index":       row number (0-indexed, excluding header),
                  "column_names":    comma-separated column names string,
                  "table_citation":  "[TB:filename:rowN]"
              }
        - Append the Document to ``self._tables[filename]``.

        After loading, print a summary: "Loaded N rows from M tables."

        Args:
            tables_dir: Path to the directory containing CSV files.

        Raises:
            FileNotFoundError: If *tables_dir* does not exist.
            ValueError: If a CSV file has no header row.

        TODO — Stop 3:
            1. Validate tables_dir exists.
            2. Iterate over ``tables_dir.glob("*.csv")``.
            3. For each file, read with ``csv.DictReader``.
            4. Build row Documents as described above.
            5. Store in ``self._tables``.
            6. Print summary.
        """
        raise NotImplementedError(
            "TODO: implement load_tables() — see Stop 3 in m2-capstone-rag-knowledge-base.md"
        )

    def retrieve(self, query: str, k: int = TABLE_RETRIEVAL_K) -> list[Document]:
        """Retrieve the *k* most relevant table rows for *query*.

        Retrieval strategy (implement at least one):

        OPTION A — Keyword matching:
            Tokenise the query and score each row by the number of matching
            column names or cell values (case-insensitive).  Sort by descending
            score.  Break ties by row index.

        OPTION B — Semantic similarity:
            Use OpenAI embeddings over the serialised row strings (same model
            as the vector store: text-embedding-3-small).  Store embeddings in
            memory after load_tables(); at query time compute cosine similarity.

        OPTION C — Numeric comparison (required for Test Case C):
            Detect superlative queries ("most", "highest", "cheapest", "most
            storage") and sort the relevant column numerically.  Return the
            top/bottom row accordingly.  This must be implemented in addition
            to one of A or B.

        Numeric comparisons must be exact (no ±tolerance) unless the brief
        specifies otherwise.  The citation in returned Documents MUST include
        the ``[TB:filename:rowN]`` format in ``metadata["table_citation"]``.

        Args:
            query: The user's question (e.g. "Which laptop has the most storage?").
            k:     Maximum number of row Documents to return.

        Returns:
            A list of at most *k* row Documents sorted by relevance descending.

        Raises:
            RuntimeError: If :meth:`load_tables` has not been called yet.

        Example::

            docs = tr.retrieve("How much does the Laptop Lite V3 cost?", k=1)
            assert "499" in docs[0].page_content
            assert "[TB:laptop_specs.csv:" in docs[0].metadata["table_citation"]

        TODO — Stop 3:
            1. Raise RuntimeError if self._tables is empty.
            2. Implement at least OPTION A and OPTION C.
            3. Collect results from all loaded tables.
            4. Sort and return top k.
        """
        raise NotImplementedError(
            "TODO: implement retrieve() — see Stop 3 in m2-capstone-rag-knowledge-base.md"
        )
