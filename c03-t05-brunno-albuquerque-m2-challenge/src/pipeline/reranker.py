"""
pipeline/reranker.py — Cross-encoder re-ranking stage.

Position in the architecture:
    vectorstore.py (MMR retriever)  →  reranker.py  →  LLM context window

Stop 2 (W5): Implement rerank().

WHY CROSS-ENCODER AFTER RETRIEVAL (for your docstring)?
    The MMR retriever uses a bi-encoder: query and document are embedded
    *separately*, then compared by cosine similarity.  This is fast (O(1) per
    query after indexing) but lossy — the embeddings compress meaning into fixed
    vectors and cannot model fine-grained query-document interactions.

    A cross-encoder processes (query, document) *together* through a transformer,
    capturing cross-attention between every query token and every document token.
    This is much more accurate at ranking relevant documents, but too slow to run
    over an entire corpus — hence we run it only over the small MMR candidate set
    (fetch_k=20 → k=6 → reranked top-3).

    This two-stage pattern (fast bi-encoder retrieval + precise cross-encoder
    re-ranking) is the standard production pattern for neural retrieval.

WHY RERANK ONLY THE TOP-6 AND NOT ALL DOCUMENTS?
    The cross-encoder is O(n) in the number of candidates (one forward pass per
    pair).  Running it over all corpus chunks would be prohibitively slow.
    MMR acts as a fast pre-filter, and the cross-encoder precision-sorts the
    survivors.  The final context window receives only RERANK_TOP_N=3 chunks
    to avoid context length inflation.
"""

from __future__ import annotations

from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RERANK_TOP_N: int = 3
"""Number of documents returned to the LLM after re-ranking.

Keeping the context to 3 chunks limits token usage and forces the cross-encoder
to select only the most relevant evidence, improving answer precision.
"""

_CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
"""HuggingFace cross-encoder model for re-ranking.

``ms-marco-MiniLM-L-6-v2`` is trained on the MS MARCO passage ranking dataset —
question-answer pairs similar to customer support queries.  It is fast (6-layer
MiniLM) and accurate enough for this use case.  For higher accuracy at the cost
of latency, consider ``cross-encoder/ms-marco-electra-base``.
"""


# Module-level singleton to avoid reloading the model on every call
_reranker = None


def rerank(
    query: str,
    docs: list[Document],
    top_n: int = RERANK_TOP_N,
) -> list[Document]:
    """Re-rank *docs* against *query* using a cross-encoder and return the top *top_n*.

    The cross-encoder (``cross-encoder/ms-marco-MiniLM-L-6-v2``) takes
    (query, document_text) pairs and outputs a relevance score.  Documents are
    sorted descending by score; only the top *top_n* are returned.

    The downstream RAG chain interface does not change: it still receives a list
    of Documents, just a smaller and more relevant one.

    #9
    WHY CROSS-ENCODING IS MORE ACCURATE THAN BI-ENCODER COSINE SIMILARITY:
    A bi-encoder embeds the query and document independently, then compares the
    resulting vectors with cosine similarity. This loses fine-grained interactions
    between query tokens and document tokens. A cross-encoder feeds both together
    into a transformer, allowing full cross-attention across every token pair.
    This captures nuanced relevance (entailment, intent matching, etc.) that
    simple vector similarity cannot, which is why it produces higher precision
    rankings (as explained in the module-level docstring above).


    Args:
        query:  The user's question string.
        docs:   Candidate documents from the MMR retriever
                (typically :data:`~src.pipeline.vectorstore.MMR_K` = 6 items).
        top_n:  Number of documents to return.  Defaults to :data:`RERANK_TOP_N`.

    Returns:
        A list of at most *top_n* Documents, sorted by decreasing relevance score.
        Each Document has a ``rerank_score`` entry added to its metadata.

    Raises:
        ValueError: If *docs* is empty or *top_n* < 1.

    Example::

        from src.pipeline.reranker import rerank

        mmr_results = retriever.invoke("warranty coverage for accidental damage")
        final_docs  = rerank("warranty coverage for accidental damage", mmr_results)
        assert len(final_docs) <= RERANK_TOP_N

    TODO — Stop 2:
        1. Validate inputs: raise ValueError if docs is empty or top_n < 1. (OK)
        2. Import ``from sentence_transformers import CrossEncoder`` inside the
           function (lazy import avoids loading the model until it is needed).  (OK)
        3. Instantiate ``CrossEncoder(_CROSS_ENCODER_MODEL)`` — consider caching
           it as a module-level singleton to avoid reloading on every call. (OK)
        4. Build pairs: ``[(query, doc.page_content) for doc in docs]``. (OK)
        5. Call ``reranker.predict(pairs)`` to get a list of float scores. (OK)
        6. Sort ``zip(scores, docs)`` descending by score. (OK)
        7. Add ``{'rerank_score': score}`` to each returned doc's metadata. (OK)
        8. Return the top *top_n* documents. (OK)
        9. Add a docstring comment explaining WHY cross-encoding is more accurate
           than bi-encoder cosine similarity (see module-level docstring above). (OK)
    """
    # 1. Input validation
    if not docs:
        raise ValueError("Cannot rerank: docs list is empty.")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    # 2. Lazy import
    from sentence_transformers import CrossEncoder

    # 3. Singleton pattern — load only once
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_CROSS_ENCODER_MODEL)

    # 4. Build (query, document) pairs
    pairs = [(query, doc.page_content) for doc in docs]

    # 5. Get relevance scores from the cross-encoder
    scores = _reranker.predict(pairs)

    # 6. Sort documents by score (descending)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    # 7 & 8. Add rerank_score to metadata and return top_n documents
    top_docs = []
    for score, doc in ranked[:top_n]:
        # Add score to metadata (create new metadata dict to avoid mutating original)
        new_metadata = dict(doc.metadata)
        new_metadata["rerank_score"] = float(score)
        top_docs.append(Document(page_content=doc.page_content, metadata=new_metadata))

    return top_docs