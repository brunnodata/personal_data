"""
graph — knowledge graph layer for entity-aware, multi-hop retrieval.

Modules:
    knowledge_graph — TechStoreKnowledgeGraph: build and query a property
                      graph from the TechStore Plus corpus (Stop 3)

WHY GRAPH RAG (for your Stop 3 docstring)?
    Dense vector search embeds every chunk into a high-dimensional space and
    retrieves by geometric proximity.  This works well for semantic similarity
    but loses *structural* relationships: the fact that 'Laptop Pro X1'
    COVERED_BY 'Premium Protection Plan' is not reliably captured by cosine
    distance alone, especially if the two concepts appear in different documents.

    A property graph explicitly encodes entity-relation-entity triples with
    provenance.  Graph traversal can follow warranty chains, policy amendments,
    and product-category hierarchies in a way that vector search cannot.

    Graph RAG complements vector retrieval: entity-dense queries (product names,
    policy references, warranty tiers) are routed through the graph; open-ended
    semantic queries fall back to the vector store.  TechStoreRAGAgent merges
    both evidence streams before generation.

    Forward reference to M3: in Module 3 you will refactor this routing logic
    into a LangGraph StateGraph where the graph retrieval node and the vector
    retrieval node are parallel branches connected by an edge condition.
"""
