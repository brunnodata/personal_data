"""
pipeline — document ingestion, embedding, and retrieval layer.

Modules:
    loader      — load and chunk the TechStore Plus corpus (Stop 1)
    vectorstore — ChromaDB setup and MMR retriever (Stop 1 + Stop 2)
    reranker    — cross-encoder re-ranking (Stop 2)
"""
