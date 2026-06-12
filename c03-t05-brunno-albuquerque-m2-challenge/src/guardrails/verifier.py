"""
guardrails/verifier.py — Claim decomposition, entailment check, and decision gate.

Position in the architecture:
    writer.py (cited answer)  →  verifier.py  →  TechStoreRAGAgent (final answer)

Stop 3 (W6): Implement verify_answer().

DECISION GATE LOGIC (implement exactly as specified):
    If claim_support_rate >= 0.85:
        return GuardrailedAnswer(decision="answer", ...)

    If claim_support_rate < 0.85 AND contradiction_rate == 0:
        return GuardrailedAnswer(decision="answer_with_disclaimer", ...)

    If contradiction_rate > 0:
        return GuardrailedAnswer(decision="extractive", answer=most_relevant_chunk)

    If no relevant evidence exists (context_docs is empty OR all claims are "unknown"):
        return GuardrailedAnswer(decision="no_answer",
                                 answer="I don't have that information in our documentation.")

WHY THESE THRESHOLDS?
    0.85 support rate: allows minor contextual inferences while blocking
    answers where fewer than 85% of claims are verifiably grounded.

    contradiction_rate > 0: even a single contradiction (LLM asserted X,
    source says not-X) triggers extractive fallback because the answer is
    actively misleading, not merely unverified.

    extractive fallback returns the verbatim most-relevant chunk so the user
    gets accurate information without the risk of a generated misstatement.
"""

from __future__ import annotations

from src.rag_agent import GuardrailedAnswer
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORT_RATE_THRESHOLD: float = 0.85
"""Minimum fraction of claims that must be supported for a clean answer.

Tuning guide:
    0.95 — very conservative; suits high-stakes legal or medical contexts.
    0.85 — balanced default for e-commerce customer support.
    0.70 — lenient; acceptable for low-risk informational queries.
"""

_LLM_MODEL: str = "gpt-4.1-mini"


def verify_answer(
    answer: str,
    context: list[Document],
) -> "GuardrailedAnswer":
    """Decompose *answer* into atomic claims, verify each against *context*, apply gate.

    Verification procedure:
    1. Use an LLM to decompose *answer* into a list of atomic claims
       (one factual statement per item, no compound claims).
    2. For each claim, use a second LLM call with an entailment prompt:
       "Does the following passage SUPPORT, CONTRADICT, or is it UNKNOWN
        regarding this claim?"
       Map the response to one of ``{"supported", "contradicted", "unknown"}``.
    3. Compute:
       - ``claim_support_rate``   = supported_count / total_claims
       - ``contradiction_rate``   = contradicted_count / total_claims
    4. Apply the decision gate (see module-level docstring).
    5. Collect source filenames from context_docs metadata into ``cited_sources``.

    The caller (:class:`~src.rag_agent.TechStoreRAGAgent`) uses the returned
    :class:`~src.rag_agent.GuardrailedAnswer` to decide what to return to the user
    and what to log for quality monitoring.

    Args:
        answer:  The raw cited answer string from :func:`~src.guardrails.writer.build_cited_answer`.
        context: The context documents used to generate *answer*.
                 Must match the sources referenced by the inline citations.

    Returns:
        A fully populated :class:`~src.rag_agent.GuardrailedAnswer` instance.

    Raises:
        ValueError: If *answer* is an empty string.
        openai.AuthenticationError: If ``OPENAI_API_KEY`` is not set.

    Example::

        result = verify_answer(raw_answer, context_docs)
        if result.decision == "no_answer":
            print("No grounded evidence found.")
        elif result.decision == "answer_with_disclaimer":
            print(result.answer + "\\n[Note: some claims could not be verified]")
        else:
            print(result.answer)

    TODO — Stop 3:
        1. Validate *answer* is non-empty.
        2. Handle the edge case: if context is empty, return immediately with
           decision="no_answer".
        3. Use ChatOpenAI to decompose *answer* into atomic claims (JSON list).
        4. For each claim, call the entailment prompt against the joined context
           text.  Map response to "supported" / "contradicted" / "unknown".
        5. Compute claim_support_rate and contradiction_rate.
        6. Identify the most relevant chunk (highest rerank_score in metadata,
           or first if not present) for the extractive fallback.
        7. Apply the decision gate and construct a GuardrailedAnswer.
        8. Extract cited_sources from context metadata (deduplicated list of
           source filenames).
        9. Return the GuardrailedAnswer.
    """
    raise NotImplementedError(
        "TODO: implement verify_answer() — see Stop 3 in m2-capstone-rag-knowledge-base.md"
    )
