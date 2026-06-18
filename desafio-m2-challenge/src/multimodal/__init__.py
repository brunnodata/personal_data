"""
multimodal — non-text modality retrieval (Stop 3).

Modules:
    table_retriever — TableRetriever: parse CSV tables and retrieve rows
                      as Documents for numeric/comparison queries.

WHY MULTIMODAL RETRIEVAL?
    Product specification queries like "How much RAM does the Laptop Max Pro have?"
    or "Which laptop has the most storage?" require exact numeric values, not
    semantic similarity.  A vector retriever trained on prose text may not surface
    the exact cell value from a spec table — it might retrieve a paragraph that
    mentions 32 GB in passing without it being the primary subject.

    TableRetriever parses CSV tables into row-level Documents and uses exact
    or fuzzy string matching to find relevant rows, then returns them with
    table citations ([TB:filename:rowN]).  This guarantees numeric accuracy
    and satisfies the Mandatory Test Case C requirement.

    Option B alternative: an ImageRetriever that stores OCR-extracted captions
    as Documents.  Both options expose the same unified retrieve() interface,
    so TechStoreRAGAgent can call either without knowing the modality.
"""
