"""
guardrails/writer.py — Citation-binding writer prompt.

Position in the architecture:
    reranker.py (top-3 docs) + graph snippets  →  writer.py  →  verifier.py

Stop 3 (W6): Implement build_cited_answer().

WHY CITATION BINDING (for your docstring)?
    Requiring inline citations ([key]) serves two purposes:
    1. User trust: readers can verify which source backs each claim.
    2. Verifiability: verifier.py uses the [key] annotations to look up the
       exact source chunk for each claim and run entailment checks.

    Without citation binding, the verifier would have to try every source chunk
    against every claim — O(claims × chunks).  With binding, it is O(claims).

    Citation format: [source_filename] for vector-store chunks,
                     [G:subject→object] for graph-sourced snippets,
                     [TB:filename:rowN] for table retrieval results.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

_LLM_MODEL: str = "gpt-4.1-mini"


def build_cited_answer(
    question: str,
    context_docs: list[Document],
) -> str:
    """Generate a cited answer grounded in *context_docs*.

    Every sentence in the answer must end with an inline citation of the form
    ``[source_filename]`` where ``source_filename`` is the ``source`` metadata
    field of the most relevant context document for that sentence.

    The writer prompt must instruct the LLM to:
    - Use ONLY information present in *context_docs*.
    - End every sentence with ``[source_filename]``.
    - If the answer to the question is not present in context_docs, output the
      exact string: ``"I don't have that information in our documentation."``
    - Never invent facts, model numbers, prices, or dates not in the context.

    Args:
        question:     The user's question.
        context_docs: The final context documents (typically top-3 from
                      :func:`~src.pipeline.reranker.rerank` plus any graph
                      snippets serialised as Documents).

    Returns:
        A string answer with inline ``[source_filename]`` citations.

    Raises:
        ValueError: If *context_docs* is empty.
        openai.AuthenticationError: If ``OPENAI_API_KEY`` is not set.

    Example::

        answer = build_cited_answer(
            "What is the return window for a refund?",
            top_docs,
        )
        # "You have 7 days from the delivery date to request a refund. [policy_return_policy.txt]"

    TODO — Stop 3:
        1. Validate context_docs is non-empty; raise ValueError if empty.
        2. Build a context string: for each doc, format as:
               "[source_filename]\\n{page_content}"
           Join with "\\n---\\n".
        3. Construct a ChatPromptTemplate:
               "You are a TechStore Plus support assistant. Answer using ONLY
               the context below. End every sentence with [source_filename].
               If the answer is not in the context, reply exactly:
               'I don't have that information in our documentation.'
               \\nContext:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:"
        4. Create ``ChatOpenAI(model=_LLM_MODEL, temperature=0)``.
        5. Invoke the chain and return the answer string.
    """
    raise NotImplementedError(
        "TODO: implement build_cited_answer() — see Stop 3 in m2-capstone-rag-knowledge-base.md"
    )
