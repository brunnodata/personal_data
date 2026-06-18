"""
evaluate_retrieval.py
Retrieval Quality Evaluation — Stop 2
Compares Baseline (similarity) vs Optimized (MMR + Cross-Encoder)
"""

from typing import List, Dict, Any
import json
import os
from dotenv import load_dotenv
load_dotenv()

# === Project imports ===
from src.pipeline.vectorstore import load_vectorstore, get_mmr_retriever
from src.pipeline.reranker import rerank
from langchain_core.documents import Document

# ============================================================================
# Evaluation Set (copied from retrieval-metrics.md)
# ============================================================================
QUERIES_AND_GT: List[Dict[str, Any]] = [
    {"q": "What is the return window for a refund?",
     "relevant": ["policy_return_policy.txt"]},
    {"q": "How do I reset the Router NX300?",
     "relevant": ["product_manual_router_nx300.txt", "support_router_wont_connect.txt"]},
    {"q": "What does the Premium Protection Plan cover?",
     "relevant": ["policy_warranty_terms.txt"]},
    {"q": "Steps to file a warranty claim online",
     "relevant": ["support_warranty_claim_process.txt"]},
    {"q": "Laptop Pro X1 specifications",
     "relevant": ["product_manual_laptop_pro_x1.txt", "laptop_specs.csv"]},
    {"q": "How do I pair a Zigbee device with the Smart Hub?",
     "relevant": ["product_manual_smart_hub_home.txt"]},
    {"q": "Does TechStore Plus cover accidental damage?",
     "relevant": ["policy_warranty_terms.txt"]},
    {"q": "Laptop won't turn on — first troubleshooting step",
     "relevant": ["support_laptop_wont_power_on.txt"]},
    {"q": "What is the restocking fee for an opened product?",
     "relevant": ["policy_return_policy.txt"]},
    {"q": "Warranty period for networking equipment",
     "relevant": ["policy_warranty_terms.txt", "product_manual_router_nx300.txt"]},
]

"""
# Unique Test Cases for Debugging
QUERIES_AND_GT: List[Dict[str, Any]] = [
    {"q": "What is the return window for a refund?",
     "relevant": ["policy_return_policy.txt"]},
]
"""

# ============================================================================
# Metrics
# ============================================================================
def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculate Precision@k."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k

def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""
    for rank, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0

def get_filenames(docs: List[Document]) -> List[str]:
    """Extract filenames from the 'source' or 'file_name' metadata field."""
    filenames = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_name") or ""
        if src:
            # Keep only the filename (remove path)
            filenames.append(src.split("/")[-1].split("\\")[-1])
    return filenames

# ============================================================================
# Evaluation Functions
# ============================================================================
def evaluate_baseline(vs, queries: List[Dict]) -> Dict[str, float]:
    """Baseline: simple similarity search with k=4."""
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prec3, prec6, mrrs = [], [], []
    for item in queries:
        docs = retriever.get_relevant_documents(item["q"])
        retrieved = get_filenames(docs)
        prec3.append(precision_at_k(retrieved, item["relevant"], 3))
        prec6.append(precision_at_k(retrieved, item["relevant"], 6))
        mrrs.append(mean_reciprocal_rank(retrieved, item["relevant"]))

    return {
        "Precision@3": round(sum(prec3) / len(prec3), 3),
        "Precision@6": round(sum(prec6) / len(prec6), 3),
        "MRR": round(sum(mrrs) / len(mrrs), 3),
    }

def evaluate_optimized(vs, queries: List[Dict]) -> Dict[str, float]:
    """Optimized: MMR (k=6, fetch_k=20) + Cross-Encoder rerank (top-3)."""
    mmr_retriever = get_mmr_retriever(vs)

    prec3, prec6, mrrs = [], [], []
    for item in queries:
        # 1. Retrieve 6 chunks using MMR
        docs = mmr_retriever.get_relevant_documents(item["q"])

        # 2. Re-rank and keep only top-3
        reranked_docs = rerank(item["q"], docs, top_n=3)
        retrieved = get_filenames(reranked_docs)

        prec3.append(precision_at_k(retrieved, item["relevant"], 3))
        prec6.append(precision_at_k(retrieved, item["relevant"], 6))
        mrrs.append(mean_reciprocal_rank(retrieved, item["relevant"]))

    return {
        "Precision@3": round(sum(prec3) / len(prec3), 3),
        "Precision@6": round(sum(prec6) / len(prec6), 3),
        "MRR": round(sum(mrrs) / len(mrrs), 3),
    }

# ============================================================================
# Main
# ============================================================================
def main():
    print("Loading vectorstore...")
    vs = load_vectorstore()

    print("\n=== Evaluating Baseline (Similarity k=4) ===")
    baseline = evaluate_baseline(vs, QUERIES_AND_GT)
    print(baseline)

    print("\n=== Evaluating Optimized (MMR k=6 + Reranker top-3) ===")
    optimized = evaluate_optimized(vs, QUERIES_AND_GT)
    print(optimized)

    print("\n=== Comparison ===")
    print(f"Baseline   → P@3: {baseline['Precision@3']}, P@6: {baseline['Precision@6']}, MRR: {baseline['MRR']}")
    print(f"Optimized  → P@3: {optimized['Precision@3']}, P@6: {optimized['Precision@6']}, MRR: {optimized['MRR']}")

if __name__ == "__main__":
    main()