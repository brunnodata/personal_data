"""
TechStore Plus RAG Knowledge Base — M2 Capstone
================================================

Top-level package for the TechStore Plus RAG system.

Public entry point:

    from src.rag_agent import TechStoreRAGAgent

    agent = TechStoreRAGAgent()
    result = agent.answer("What is the return policy for laptops?")
    print(result.answer)

Build progression:
    Stop 1 (W4) — src/pipeline/loader.py, vectorstore.py  → basic RAG
    Stop 2 (W5) — src/pipeline/reranker.py                → MMR + re-ranking
    Stop 3 (W6) — src/graph/, src/guardrails/,
                  src/multimodal/, src/rag_agent.py        → production system
"""
