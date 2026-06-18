"""
rag_agent.py — TechStoreRAGAgent: public entry point for the M2 capstone.

This module is the top of the import hierarchy.  All other src/ modules
import from langchain_*, not from this module, to avoid circular imports.

Stop 3 (W6): Implement TechStoreRAGAgent.answer().

Architecture overview:

    User question
        │
        ▼
    TechStoreRAGAgent.answer()
        │
        ├── [always] pipeline/vectorstore → pipeline/reranker   (MMR + cross-encoder)
        │
        ├── [entity-dense] graph/knowledge_graph.query_subgraph  (Graph RAG)
        │
        ├── [numeric/table] multimodal/table_retriever.retrieve  (Table grounding)
        │
        ├── [merge evidence]
        │
        ├── guardrails/writer.build_cited_answer                 (citation binding)
        │
        └── guardrails/verifier.verify_answer                    (decision gate)
                │
                └── GuardrailedAnswer  →  caller

M1 integration (forward reference from Stop 3 Component 4):
    The TechStoreRAGAgent replaces hard-coded product lookups in the M1 MemoryAgent:

        from src.rag_agent import TechStoreRAGAgent

        rag = TechStoreRAGAgent()

        @tool
        def search_knowledge_base(question: str) -> str:
            \"""Search TechStore Plus product manuals, support articles, and policies.\"""
            result = rag.answer(question)
            return result.answer

    This integration works without modifying MemoryAgent because answer() always
    returns a GuardrailedAnswer with a non-empty ``answer`` field (even for no_answer
    decisions, the field contains the safe fallback string).

Forward reference to M3 (LangGraph):
    In Module 3, the manual routing logic inside answer() — the if/else branches
    for entity-dense vs. semantic vs. table queries — will be replaced by a
    LangGraph StateGraph where:
    - Each retrieval path (vector, graph, multimodal) is a ToolNode.
    - The routing logic is an edge condition function.
    - The decision gate is a conditional edge.
    Keeping the components loosely coupled in this module makes that refactor
    straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# GuardrailedAnswer — fully implemented (used by guardrails/verifier.py)
# ---------------------------------------------------------------------------

@dataclass
class GuardrailedAnswer:
    """The return type of :meth:`TechStoreRAGAgent.answer`.

    Carries both the answer text and the guardrail metadata so callers can
    log quality signals, display disclaimers, or route to human escalation.

    Attributes:
        answer: The final answer string to present to the user.
            - For ``decision="answer"`` or ``"answer_with_disclaimer"``: the
              generated answer with inline citations (``[source_filename]``).
            - For ``decision="extractive"``: the verbatim most-relevant chunk.
            - For ``decision="no_answer"``: the exact string
              ``"I don't have that information in our documentation."``.

        decision: One of four values controlling how the answer was produced:
            - ``"answer"`` — claim_support_rate >= 0.85, no contradictions.
            - ``"answer_with_disclaimer"`` — claim_support_rate < 0.85 but
              no contradictions; answer is partially supported.
            - ``"extractive"`` — contradiction_rate > 0; verbatim chunk returned.
            - ``"no_answer"`` — no relevant evidence in the corpus.

        claim_support_rate: Fraction of atomic claims supported by the context.
            Range [0.0, 1.0].  0.0 for no_answer decisions.

        contradiction_rate: Fraction of atomic claims contradicted by the context.
            Range [0.0, 1.0].  0.0 for no_answer decisions.

        cited_sources: Deduplicated list of source identifiers referenced in the
            answer.  Format examples:
            - ``"policy_return_policy.txt"``   (vector store source)
            - ``"[G:Laptop Pro X1->Premium Protection Plan]"`` (graph edge)
            - ``"[TB:laptop_specs.csv:row0]"`` (table row)

    Example::

        result = agent.answer("What is the return period for a refund?")
        assert result.decision in ("answer", "answer_with_disclaimer",
                                   "extractive", "no_answer")
        assert 0.0 <= result.claim_support_rate <= 1.0
        assert isinstance(result.cited_sources, list)
    """

    answer: str
    decision: str  # "answer" | "answer_with_disclaimer" | "extractive" | "no_answer"
    claim_support_rate: float
    contradiction_rate: float
    cited_sources: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        valid_decisions = {"answer", "answer_with_disclaimer", "extractive", "no_answer"}
        if self.decision not in valid_decisions:
            raise ValueError(
                f"decision must be one of {valid_decisions}, got {self.decision!r}"
            )
        if not (0.0 <= self.claim_support_rate <= 1.0):
            raise ValueError(
                f"claim_support_rate must be in [0.0, 1.0], got {self.claim_support_rate}"
            )
        if not (0.0 <= self.contradiction_rate <= 1.0):
            raise ValueError(
                f"contradiction_rate must be in [0.0, 1.0], got {self.contradiction_rate}"
            )


# ---------------------------------------------------------------------------
# TechStoreRAGAgent
# ---------------------------------------------------------------------------

class TechStoreRAGAgent:
    """Production-quality RAG agent for TechStore Plus knowledge queries.

    Integrates four retrieval and generation components:
    1. Vector retrieval: MMR + cross-encoder re-ranking (Stop 2 pipeline).
    2. Graph RAG: entity-aware multi-hop traversal (Stop 3).
    3. Multimodal retrieval: CSV table grounding (Stop 3).
    4. Hallucination guardrails: citation binding + decision gate (Stop 3).

    Usage::

        agent = TechStoreRAGAgent()
        result = agent.answer("What is TechStore Plus's return policy?")
        print(result.answer)
        print(result.decision)         # "answer" | "answer_with_disclaimer" | ...
        print(result.claim_support_rate)

    M1 integration (as a LangChain @tool)::

        from src.rag_agent import TechStoreRAGAgent
        from langchain_core.tools import tool

        rag = TechStoreRAGAgent()

        @tool
        def search_knowledge_base(question: str) -> str:
            \"""Search TechStore Plus product manuals, support articles, and policies.\"""
            result = rag.answer(question)
            return result.answer

    Attributes:
        _vectorstore: Loaded ChromaDB vectorstore instance (initialised lazily).
        _kg:          TechStoreKnowledgeGraph instance (initialised lazily).
        _table_retriever: TableRetriever instance (initialised lazily).

    Note:
        Components are initialised lazily on first call to answer() to avoid
        loading models at import time.  The vectorstore and knowledge graph
        are built once and reused across calls.
    """

    def __init__(self) -> None:
        self._vectorstore = None
        self._kg = None
        self._table_retriever = None

    def _ensure_initialized(self) -> None:
        """Lazy-initialise all components.

        TODO — Stop 3:
            1. If self._vectorstore is None, call load_vectorstore() from
               src.pipeline.vectorstore.  If the chroma_db/ directory does not
               exist, call build_vectorstore(chunk_documents(load_documents()))
               to build it from scratch.
            2. If self._kg is None, instantiate TechStoreKnowledgeGraph and call
               extract_and_build(load_documents()).
            3. If self._table_retriever is None, instantiate TableRetriever and
               call load_tables().
        """
        raise NotImplementedError(
            "TODO: implement _ensure_initialized() — see Stop 3 in m2-capstone-rag-knowledge-base.md"
        )

    def _is_entity_dense(self, question: str) -> bool:
        """Return True if *question* contains entity-dense signals.

        Entity-dense signals include product model names, warranty tier names,
        policy version references, and named policy documents.

        Used by answer() to decide whether to invoke Graph RAG in addition to
        the vector retriever.

        Args:
            question: The user's question string.

        Returns:
            True if the question likely benefits from graph traversal.

        TODO — Stop 3:
            1. Define a list of entity keywords: product names from laptop_specs.csv,
               warranty tier names ("basic", "standard", "premium", "extended",
               "protection plan"), policy references ("2024", "2023", "POL-").
            2. Return True if any keyword appears in the question (case-insensitive).
            3. You may optionally use an LLM call for more nuanced detection.
        """
        raise NotImplementedError(
            "TODO: implement _is_entity_dense() — see Stop 3 in m2-capstone-rag-knowledge-base.md"
        )

    def _is_table_query(self, question: str) -> bool:
        """Return True if *question* targets numeric or table data.

        Table query signals: superlatives ("most", "cheapest", "highest",
        "largest"), explicit comparisons ("compare", "all models"), or direct
        numeric attribute questions ("how much RAM", "what price", "storage size").

        Args:
            question: The user's question string.

        Returns:
            True if the question likely benefits from table retrieval.

        TODO — Stop 3:
            1. Define a list of table-query keywords (superlatives, comparisons,
               numeric attribute names from the CSV column headers).
            2. Return True if any keyword appears in the question (case-insensitive).
        """
        raise NotImplementedError(
            "TODO: implement _is_table_query() — see Stop 3 in m2-capstone-rag-knowledge-base.md"
        )

    def answer(
        self,
        question: str,
        context: Optional[dict] = None,
    ) -> GuardrailedAnswer:
        """Answer *question* using the full TechStore Plus RAG pipeline.

        Routing logic:
        1. Always: MMR retrieval → cross-encoder re-ranking → top-3 vector docs.
        2. If entity-dense: Graph RAG → query_subgraph → serialised snippets.
        3. If table query: TableRetriever → matching rows → table doc(s).
        4. Merge all evidence into a single context list.
        5. Citation-binding writer → raw cited answer string.
        6. Claim verifier + decision gate → GuardrailedAnswer.

        The optional *context* dict may contain customer metadata from the M1
        MemoryAgent (e.g., ``{"email": "user@company.com", "recent_intent": "returns"}``).
        Use it to add a filter hint to the writer prompt (e.g., "this customer
        asked about returns — prioritise policy documents") or to the graph
        seed entity list.

        Args:
            question: The user's question.
            context:  Optional dict with caller metadata (from M1 MemoryAgent).

        Returns:
            A :class:`GuardrailedAnswer` instance.

        Raises:
            RuntimeError: If initialisation fails (e.g., missing OPENAI_API_KEY,
                          missing chroma_db/ directory and no data/ documents).

        Example::

            agent = TechStoreRAGAgent()

            # On-topic query — should return decision="answer"
            r1 = agent.answer("What is the return period for a refund?")
            assert "7" in r1.answer          # 7-day refund window
            assert r1.decision in ("answer", "answer_with_disclaimer")

            # Off-topic query — should return decision="no_answer"
            r2 = agent.answer("What is the capital of France?")
            assert r2.decision == "no_answer"

        TODO — Stop 3:
            1. Call self._ensure_initialized().
            2. Run MMR retrieval: get_mmr_retriever(self._vectorstore).invoke(question).
            3. Run rerank(question, mmr_docs).
            4. If self._is_entity_dense(question):
               a. Extract seed entities from the question (simple keyword match
                  against known entity list, or use an LLM).
               b. Call self._kg.query_subgraph(seed_entities).
               c. Serialise each snippet as a Document and append to context_docs.
                  Set metadata["source"] = "[G:{subject}->{object}]" for citation.
            5. If self._is_table_query(question):
               a. Call self._table_retriever.retrieve(question).
               b. Append to context_docs.
            6. If context_docs is empty after steps 2-5, return immediately:
               GuardrailedAnswer(answer="I don't have that information in our documentation.",
                                 decision="no_answer", claim_support_rate=0.0,
                                 contradiction_rate=0.0, cited_sources=[]).
            7. Call build_cited_answer(question, context_docs).
            8. Call verify_answer(raw_answer, context_docs).
            9. Return the GuardrailedAnswer.
        """
        raise NotImplementedError(
            "TODO: implement answer() — see Stop 3 in m2-capstone-rag-knowledge-base.md"
        )
