"""
graph/knowledge_graph.py — Lightweight property graph over the TechStore Plus corpus.

Position in the architecture:
    loader.py  →  knowledge_graph.py  →  TechStoreRAGAgent (query routing)

Stop 3 (W6): Implement TechStoreKnowledgeGraph.

WHY GRAPH RAG?
    Dense vector search finds semantically similar chunks but loses structural
    relationships between entities.  Querying "Which products are covered under
    the extended warranty?" requires knowing:
        Laptop Pro X1  --COVERED_BY-->  Premium Protection Plan
        Router NX300   --COVERED_BY-->  Standard Warranty
        Smart Hub Home --COVERED_BY-->  Premium Protection Plan

    These triples can be extracted from the corpus by an LLM and stored in a
    networkx DiGraph.  Multi-hop traversal (up to 2 hops) then surfaces
    warranty chains, policy amendments, and product-category hierarchies that
    a simple cosine search would miss.

    TechStoreRAGAgent uses graph retrieval when the question contains
    entity-dense signals (product model names, warranty tier names, policy
    version references) and falls back to the vector store for open semantic
    questions.

RELATION ALLOWLIST (to restrict traversal noise):
    COVERED_BY     — product covered by a warranty tier or policy
    PART_OF        — component or accessory belonging to a product family
    AMENDS         — a policy document that supersedes or modifies another
    APPLIES_TO     — a policy or service term that applies to a product category
    REQUIRES       — a product or service that requires another product/service
    SUPERSEDED_BY  — an older policy document replaced by a newer one

Forward reference to M3 (LangGraph):
    In Module 3 you will replace the manual if/else routing inside
    TechStoreRAGAgent.answer() with a LangGraph StateGraph where this graph
    retrieval path is a dedicated ToolNode and the routing decision is an
    edge condition.  Keep the components clean and the interface stable.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HOPS: int = 2
"""Default number of hops for subgraph traversal.

2 hops is sufficient for most TechStore Plus queries:
    Product → COVERED_BY → WarrantyTier → APPLIES_TO → DamageType
"""

DEFAULT_RELATION_ALLOWLIST: list[str] = [
    "COVERED_BY",
    "PART_OF",
    "AMENDS",
    "APPLIES_TO",
    "REQUIRES",
    "SUPERSEDED_BY",
]
"""Relations that are traversed during subgraph expansion.

A narrow allowlist prevents the traversal from following noisy or
low-confidence triples extracted by the LLM.
"""


class TechStoreKnowledgeGraph:
    """In-memory property graph for entity-aware retrieval over TechStore Plus docs.

    The graph is a directed networkx ``DiGraph``.  Nodes represent entities
    (products, policies, services, concepts).  Edges represent typed relations
    with provenance metadata.

    Typical workflow::

        kg = TechStoreKnowledgeGraph()
        kg.extract_and_build(chunks)                       # Stop 3
        snippets = kg.query_subgraph(["Laptop Pro X1"])    # Stop 3

    Attributes:
        graph: The underlying :class:`networkx.DiGraph` instance.

    Note:
        This is an in-memory graph — it is rebuilt on every run.  Persistence
        (serialising to JSON or a graph database) is a known limitation
        documented in the README.
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    def add_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        source_id: str,
        quote: str,
        as_of: Optional[str] = None,
    ) -> None:
        """Add a provenance-tracked directed edge to the graph.

        If *subject* or *obj* nodes do not already exist, they are created
        with default attributes.  If the edge already exists, its metadata
        is updated (later triples win on conflicting fields).

        Node attributes set (or updated):
            ``type``  — inferred from ``relation`` or left as ``"Entity"``

        Edge attributes:
            ``relation``  — the relation label (e.g. ``"COVERED_BY"``)
            ``source_id`` — filename of the document this triple came from
            ``quote``     — verbatim short excerpt supporting this triple
            ``as_of``     — optional ISO-8601 date string (e.g. ``"2024-01-01"``)
                            used for temporal policy queries

        Args:
            subject:   Subject entity name (e.g. ``"Laptop Pro X1"``).
            relation:  Relation type from the allowlist (e.g. ``"COVERED_BY"``).
            obj:       Object entity name (e.g. ``"Premium Protection Plan"``).
            source_id: Source document filename (e.g. ``"policy_warranty_terms.txt"``).
            quote:     Short verbatim quote from *source_id* supporting this triple.
            as_of:     Optional date this triple became effective (ISO-8601 string).

        Example::

            kg.add_triple(
                subject="Laptop Pro X1",
                relation="COVERED_BY",
                obj="Premium Protection Plan",
                source_id="policy_warranty_terms.txt",
                quote="Premium Protection Plan covers the Laptop Pro X1 for 36 months",
                as_of="2024-01-01",
            )

        TODO — Stop 3:
            1. Add subject node if not present: ``self.graph.add_node(subject, type="Entity")``.
            2. Add obj node if not present.
            3. Add the directed edge with all metadata attributes.
            4. If the edge already exists, update its attributes dict.
        """
        raise NotImplementedError(
            "TODO: implement add_triple() — see Stop 3 in m2-capstone-rag-knowledge-base.md"
        )

    def extract_and_build(self, documents: list[Document]) -> None:
        """Extract entity triples from *documents* using an LLM and populate the graph.

        Uses a structured ``ChatOpenAI`` call to decompose each document chunk
        into a list of (subject, relation, object, quote) tuples.

        Prompt guidance:
            - Instruct the LLM to extract only triples where relation is in
              :data:`DEFAULT_RELATION_ALLOWLIST`.
            - Ask for a short verbatim quote from the source text for each triple.
            - Use JSON output mode or a Pydantic schema to enforce structure.

        Args:
            documents: Loaded (and optionally chunked) Document objects.
                       Passing the full un-chunked docs is acceptable here
                       because triple extraction operates at document level.

        Returns:
            None — the graph is mutated in place.

        Raises:
            openai.AuthenticationError: If ``OPENAI_API_KEY`` is missing.

        Example::

            kg = TechStoreKnowledgeGraph()
            kg.extract_and_build(load_documents())
            print(f"Graph has {kg.graph.number_of_nodes()} nodes, "
                  f"{kg.graph.number_of_edges()} edges")

        TODO — Stop 3:
            1. Set up a ``ChatOpenAI(model="gpt-4.1-mini", temperature=0)`` client.
            2. For each document, send a structured extraction prompt requesting
               triples in JSON format: ``[{subject, relation, object, quote}, ...]``.
            3. Parse the response and call ``self.add_triple()`` for each valid triple.
            4. Skip triples where ``relation`` is not in DEFAULT_RELATION_ALLOWLIST.
            5. Optionally deduplicate triples by (subject, relation, object) key.
            6. Print a summary: "Extracted N triples from M documents."
        """
        raise NotImplementedError(
            "TODO: implement extract_and_build() — see Stop 3 in m2-capstone-rag-knowledge-base.md"
        )

    def query_subgraph(
        self,
        seed_entities: list[str],
        hops: int = DEFAULT_HOPS,
        relation_allowlist: Optional[list[str]] = None,
    ) -> list[dict]:
        """Expand from *seed_entities* and return ranked supporting snippets.

        Starting from each seed entity node, performs a breadth-first traversal
        up to *hops* hops deep.  Only edges whose ``relation`` attribute is in
        *relation_allowlist* are followed.

        Each returned snippet is a dict with the following keys:
            ``subject``   — source entity name
            ``relation``  — edge relation label
            ``object``    — target entity name
            ``source_id`` — provenance document filename
            ``quote``     — verbatim supporting excerpt
            ``as_of``     — effective date string, or None
            ``hop``       — distance from the nearest seed entity (1-indexed)

        Snippets are sorted by ascending ``hop`` (closer = more relevant).
        Duplicate (subject, relation, object) triples are deduplicated.

        Args:
            seed_entities:     Entity names to start traversal from.
                               Case-insensitive match attempted if exact match fails.
            hops:              Maximum traversal depth.  Defaults to :data:`DEFAULT_HOPS`.
            relation_allowlist: Relations to follow.  Defaults to
                               :data:`DEFAULT_RELATION_ALLOWLIST`.

        Returns:
            A list of snippet dicts, sorted by hop distance ascending.

        Raises:
            ValueError: If the graph has not been populated (no nodes).

        Example::

            snippets = kg.query_subgraph(["Laptop Pro X1"], hops=2)
            for s in snippets:
                print(f"{s['subject']} --{s['relation']}--> {s['object']} "
                      f"[{s['source_id']}]")

        TODO — Stop 3:
            1. Validate the graph is non-empty.
            2. For each seed, find the matching node (exact, then case-insensitive).
            3. Use ``nx.bfs_edges(self.graph, seed, depth_limit=hops)`` or
               manual BFS to collect edges within hop budget.
            4. Filter by relation_allowlist.
            5. Build snippet dicts from edge attributes.
            6. Deduplicate and sort by hop.
            7. Return the list.
        """
        raise NotImplementedError(
            "TODO: implement query_subgraph() — see Stop 3 in m2-capstone-rag-knowledge-base.md"
        )
