"""
tests/test_mandatory_cases.py — Mandatory Test Cases for the M2 Capstone (Stop 3).

These three test cases define the passing threshold for the Functionality score:
all three must pass for Functionality to exceed 50% (20/40 pts).

Tests are marked @pytest.mark.skip until TechStoreRAGAgent is implemented.
Remove the @pytest.mark.skip decorator from each test when you are ready to run it.

Running:
    # Run all mandatory cases (skipped until implemented)
    pytest tests/test_mandatory_cases.py -v

    # Run a specific test (after removing @skip)
    pytest tests/test_mandatory_cases.py::test_case_a_no_answer_guardrail -v

    # Run with detailed output including print statements
    pytest tests/test_mandatory_cases.py -v -s

Prerequisites before running:
    1. Complete Stop 3 implementation in src/
    2. Have data/ documents loaded and chroma_db/ built (or let the agent build it)
    3. Set OPENAI_API_KEY in .env
    4. pip install -r requirements.txt
"""

import re
import pytest

# ---------------------------------------------------------------------------
# Test Case A — Decision Gate: No-Answer Guardrail
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Implement TechStoreRAGAgent first — remove skip when Stop 3 is complete")
def test_case_a_no_answer_guardrail():
    """Test Case A: Decision gate prevents hallucination when no relevant evidence exists.

    Evaluator steps (from m2-capstone-rag-knowledge-base.md):
    1. Instantiate TechStoreRAGAgent().
    2. Call agent.answer("What is the capital of France?").
    3. Verify result.decision == "no_answer" AND answer does not contain a city name.
    4. Call agent.answer("What is TechStore Plus's return period for laptops?").
    5. Verify result.decision in ("answer", "answer_with_disclaimer") AND
       answer contains a number (days).

    Pass condition:
        Off-topic queries → no_answer
        On-topic queries  → grounded answer with citation(s) and a number
    """
    from src.rag_agent import TechStoreRAGAgent, GuardrailedAnswer

    agent = TechStoreRAGAgent()

    # --- A1: Off-topic query must return no_answer ---
    result_off_topic = agent.answer("What is the capital of France?")

    assert isinstance(result_off_topic, GuardrailedAnswer), (
        "answer() must return a GuardrailedAnswer instance"
    )
    assert result_off_topic.decision == "no_answer", (
        f"Expected decision='no_answer' for off-topic query, got {result_off_topic.decision!r}. "
        f"Answer was: {result_off_topic.answer!r}"
    )
    # The corpus has no geography content — answer must not name a city
    geography_cities = ["paris", "london", "berlin", "madrid", "rome"]
    answer_lower = result_off_topic.answer.lower()
    for city in geography_cities:
        assert city not in answer_lower, (
            f"Hallucinated city '{city}' found in no_answer response: {result_off_topic.answer!r}"
        )

    # --- A2: On-topic query must return a grounded answer ---
    result_on_topic = agent.answer(
        "What is TechStore Plus's return period for laptops?"
    )

    assert isinstance(result_on_topic, GuardrailedAnswer)
    assert result_on_topic.decision in ("answer", "answer_with_disclaimer"), (
        f"Expected answer or answer_with_disclaimer for on-topic query, "
        f"got {result_on_topic.decision!r}. Answer: {result_on_topic.answer!r}"
    )
    # Answer must contain a number (number of days)
    assert re.search(r"\d+", result_on_topic.answer), (
        f"Expected a number (days) in the answer, got: {result_on_topic.answer!r}"
    )
    # Must have at least one cited source
    assert len(result_on_topic.cited_sources) >= 1, (
        f"Expected at least one cited source, got: {result_on_topic.cited_sources}"
    )


# ---------------------------------------------------------------------------
# Test Case B — Graph RAG: Entity Traversal
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Implement TechStoreRAGAgent first — remove skip when Stop 3 is complete")
def test_case_b_graph_rag_entity_traversal():
    """Test Case B: Entity-dense queries trigger Graph RAG and return provenance.

    Evaluator steps:
    1. Call agent.answer("Which products are covered under the extended warranty?").
    2. Verify cited_sources includes at least one graph-sourced citation
       (format: "[G:..." or any metadata marker distinguishing graph from vector).
    3. Call agent.answer("Does the laptop warranty cover accidental damage?").
    4. Verify answer confirms or denies coverage with a citation to the warranty doc.

    Pass condition:
        B1 — at least one graph-sourced citation in cited_sources
        B2 — grounded answer with warranty document citation
    """
    from src.rag_agent import TechStoreRAGAgent, GuardrailedAnswer

    agent = TechStoreRAGAgent()

    # --- B1: Extended warranty query — must surface graph citations ---
    result_b1 = agent.answer(
        "Which products are covered under the extended warranty?"
    )

    assert isinstance(result_b1, GuardrailedAnswer)
    assert result_b1.decision != "no_answer", (
        f"Extended warranty query should have evidence in the corpus. "
        f"Got no_answer. Check that policy_warranty_terms.txt is loaded."
    )

    # At least one cited source must originate from the knowledge graph
    # Graph citations are prefixed with "[G:" per the architecture spec
    graph_citations = [s for s in result_b1.cited_sources if s.startswith("[G:")]
    assert len(graph_citations) >= 1, (
        f"Expected at least one graph citation ([G:...]) in cited_sources. "
        f"Got: {result_b1.cited_sources}. "
        f"Ensure TechStoreKnowledgeGraph.extract_and_build() runs and "
        f"query_subgraph() finds warranty-related triples."
    )

    # --- B2: Accidental damage query — grounded with warranty doc citation ---
    result_b2 = agent.answer(
        "Does the laptop warranty cover accidental damage?"
    )

    assert isinstance(result_b2, GuardrailedAnswer)
    assert result_b2.decision in ("answer", "answer_with_disclaimer"), (
        f"Warranty coverage query should return a grounded answer, got: {result_b2.decision!r}"
    )
    # Answer must include a citation to the warranty policy document
    warranty_citations = [
        s for s in result_b2.cited_sources
        if "warranty" in s.lower() or s.startswith("[G:")
    ]
    assert len(warranty_citations) >= 1, (
        f"Expected a warranty document citation. Got cited_sources: {result_b2.cited_sources}"
    )


# ---------------------------------------------------------------------------
# Test Case C — Multimodal Retrieval: Table Grounding
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Implement TechStoreRAGAgent first — remove skip when Stop 3 is complete")
def test_case_c_multimodal_table_grounding():
    """Test Case C: Numeric queries retrieve exact values from product spec tables.

    Evaluator steps:
    1. Ensure data/tables/laptop_specs.csv contains model names, RAM, storage, price.
    2. Call agent.answer("How much RAM does the Laptop Pro X1 have?").
    3. Verify answer contains exact RAM value (16) and a table citation ([TB:...]).
    4. Call agent.answer("Which laptop has the most storage?").
    5. Verify answer names the correct model and includes a table citation.

    Pass condition:
        Both table queries return exact values with [TB:...] citations.
        No numeric hallucination allowed.

    Data assumptions (from data/tables/laptop_specs.csv):
        Laptop Pro X1:  16 GB RAM, 512 GB storage
        Laptop Air S2:   8 GB RAM, 256 GB storage
        Laptop Max Pro: 32 GB RAM, 1024 GB storage
        Laptop Lite V3:  4 GB RAM, 128 GB storage
        WorkStation Z:  64 GB RAM, 2048 GB storage

    Most storage = WorkStation Z (2048 GB / 2 TB)
    """
    from src.rag_agent import TechStoreRAGAgent, GuardrailedAnswer

    agent = TechStoreRAGAgent()

    # --- C1: Specific model RAM query ---
    result_c1 = agent.answer("How much RAM does the Laptop Pro X1 have?")

    assert isinstance(result_c1, GuardrailedAnswer)
    assert result_c1.decision != "no_answer", (
        "RAM query for Laptop Pro X1 should find data in laptop_specs.csv. "
        "Ensure TableRetriever.load_tables() has been called."
    )
    # Answer must contain the exact RAM value
    assert "16" in result_c1.answer, (
        f"Expected '16' (GB RAM) in answer for Laptop Pro X1. Got: {result_c1.answer!r}"
    )
    # Must have a table citation
    table_citations = [s for s in result_c1.cited_sources if "[TB:" in s]
    assert len(table_citations) >= 1, (
        f"Expected at least one table citation ([TB:...]) in cited_sources. "
        f"Got: {result_c1.cited_sources}"
    )

    # --- C2: Superlative / comparison query ---
    result_c2 = agent.answer("Which laptop has the most storage?")

    assert isinstance(result_c2, GuardrailedAnswer)
    assert result_c2.decision != "no_answer", (
        "Most-storage query should find data in laptop_specs.csv."
    )
    # The correct answer is WorkStation Z (2048 GB)
    # Accept "WorkStation Z" or "WorkStation" as a match
    assert re.search(r"workstation\s*z", result_c2.answer, re.IGNORECASE), (
        f"Expected 'WorkStation Z' (2048 GB storage) as the answer. "
        f"Got: {result_c2.answer!r}. "
        f"Ensure TableRetriever handles superlative comparison queries."
    )
    # Must have a table citation
    table_citations_c2 = [s for s in result_c2.cited_sources if "[TB:" in s]
    assert len(table_citations_c2) >= 1, (
        f"Expected a table citation for the most-storage query. "
        f"Got: {result_c2.cited_sources}"
    )


# ---------------------------------------------------------------------------
# Bonus: Smoke test — GuardrailedAnswer dataclass validation
# ---------------------------------------------------------------------------

def test_guardrailed_answer_valid_construction():
    """Smoke test: GuardrailedAnswer rejects invalid decision values and rates.

    This test does NOT require TechStoreRAGAgent to be implemented.
    It verifies the dataclass validation logic that is already implemented
    in src/rag_agent.py.
    """
    from src.rag_agent import GuardrailedAnswer

    # Valid construction
    ga = GuardrailedAnswer(
        answer="The return window is 7 days. [policy_return_policy.txt]",
        decision="answer",
        claim_support_rate=0.95,
        contradiction_rate=0.0,
        cited_sources=["policy_return_policy.txt"],
    )
    assert ga.decision == "answer"
    assert ga.claim_support_rate == 0.95
    assert len(ga.cited_sources) == 1

    # All four valid decisions
    for decision in ("answer", "answer_with_disclaimer", "extractive", "no_answer"):
        ga = GuardrailedAnswer(
            answer="test",
            decision=decision,
            claim_support_rate=0.5,
            contradiction_rate=0.0,
        )
        assert ga.decision == decision

    # Invalid decision raises ValueError
    with pytest.raises(ValueError, match="decision must be one of"):
        GuardrailedAnswer(
            answer="test",
            decision="hallucinated",
            claim_support_rate=0.5,
            contradiction_rate=0.0,
        )

    # Out-of-range support rate raises ValueError
    with pytest.raises(ValueError, match="claim_support_rate"):
        GuardrailedAnswer(
            answer="test",
            decision="answer",
            claim_support_rate=1.5,
            contradiction_rate=0.0,
        )
