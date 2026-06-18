"""
evaluate_retrieval.py
Retrieval Quality Evaluation — Stop 2

Compares two retrieval pipelines:
1. Baseline: Simple similarity search (k=4)
2. Optimized: MMR (k=6, fetch_k=20) + cross-encoder re-ranking (top-3)

Metrics: Precision@3, Precision@6, and Mean Reciprocal Rank (MRR)
"""

from typing import List, Dict, Any

from langchain_core.documents import Document

from src.pipeline.vectorstore import load_vectorstore, get_mmr_retriever
from src.pipeline.reranker import rerank


# ============================================================================
# Evaluation Set (from docs/retrieval-metrics.md)
# ============================================================================
QUERIES_AND_GT: List[Dict[str, Any]] = [
    {
        "q": "What is the return window for a refund?",
        "relevant": ["policy_return_policy.txt"],
    },
    {
        "q": "How do I reset the Router NX300?",
        "relevant": ["product_manual_router_nx300.txt", "support_router_wont_connect.txt"],
    },
    {
        "q": "What does the Premium Protection Plan cover?",
        "relevant": ["policy_warranty_terms.txt"],
    },
    {
        "q": "Steps to file a warranty claim online",
        "relevant": ["support_warranty_claim_process.txt"],
    },
    {
        "q": "Laptop Pro X1 specifications",
        "relevant": ["product_manual_laptop_pro_x1.txt", "laptop_specs.csv"],
    },
    {
        "q": "How do I pair a Zigbee device with the Smart Hub?",
        "relevant": ["product_manual_smart_hub_home.txt"],
    },
    {
        "q": "Does TechStore Plus cover accidental damage?",
        "relevant": ["policy_warranty_terms.txt"],
    },
    {
        "q": "Laptop won't turn on — first troubleshooting step",
        "relevant": ["support_laptop_wont_power_on.txt"],
    },
    {
        "q": "What is the restocking fee for an opened product?",
        "relevant": ["policy_return_policy.txt"],
    },
    {
        "q": "Warranty period for networking equipment",
        "relevant": ["policy_warranty_terms.txt", "product_manual_router_nx300.txt"],
    },
]


# ============================================================================
# Metric Functions
# ============================================================================
def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculate Precision@k (fraction of top-k results that are relevant)."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR): 1/rank of first relevant result."""
    for rank, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def get_filenames(docs: List[Document]) -> List[str]:
    """Extract filenames from the 'source' metadata field."""
    filenames = []
    for doc in docs:
        # Try 'source' first, fallback to 'file_name'
        source = doc.metadata.get("source") or doc.metadata.get("file_name") or ""
        if source:
            # Keep only the filename (remove directory paths)
            filename = source.split("/")[-1].split("\\")[-1]
            filenames.append(filename)
    return filenames


# ============================================================================
# Evaluation Functions
# ============================================================================
def evaluate_baseline(vs, queries: List[Dict]) -> Dict[str, float]:
    """
    Evaluate baseline retrieval: simple similarity search with k=4.

    Args:
        vs: Loaded Chroma vectorstore instance.
        queries: List of evaluation queries with ground truth.

    Returns:
        Dictionary with Precision@3, Precision@6, and MRR metrics.
    """
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prec3_scores = []
    prec6_scores = []
    mrr_scores = []

    for item in queries:
        docs = retriever.invoke(item["q"])
        retrieved = get_filenames(docs)

        prec3_scores.append(precision_at_k(retrieved, item["relevant"], 3))
        prec6_scores.append(precision_at_k(retrieved, item["relevant"], 6))
        mrr_scores.append(mean_reciprocal_rank(retrieved, item["relevant"]))

    return {
        "Precision@3": round(sum(prec3_scores) / len(prec3_scores), 3),
        "Precision@6": round(sum(prec6_scores) / len(prec6_scores), 3),
        "MRR": round(sum(mrr_scores) / len(mrr_scores), 3),
    }


def evaluate_optimized(vs, queries: List[Dict]) -> Dict[str, float]:
    """
    Evaluate optimized retrieval: MMR (k=6, fetch_k=20) + cross-encoder re-ranking (top-3).

    Args:
        vs: Loaded Chroma vectorstore instance.
        queries: List of evaluation queries with ground truth.

    Returns:
        Dictionary with Precision@3, Precision@6, and MRR metrics.
    """
    mmr_retriever = get_mmr_retriever(vs)

    prec3_scores = []
    prec6_scores = []
    mrr_scores = []

    for item in queries:
        # Stage 1: Retrieve 6 diverse chunks using MMR
        docs = mmr_retriever.invoke(item["q"])

        # Stage 2: Re-rank and keep only top-3 most relevant
        reranked_docs = rerank(item["q"], docs, top_n=3)
        retrieved = get_filenames(reranked_docs)

        prec3_scores.append(precision_at_k(retrieved, item["relevant"], 3))
        prec6_scores.append(precision_at_k(retrieved, item["relevant"], 6))
        mrr_scores.append(mean_reciprocal_rank(retrieved, item["relevant"]))

    return {
        "Precision@3": round(sum(prec3_scores) / len(prec3_scores), 3),
        "Precision@6": round(sum(prec6_scores) / len(prec6_scores), 3),
        "MRR": round(sum(mrr_scores) / len(mrr_scores), 3),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    """Load vectorstore and evaluate both pipelines."""
    print("Loading vectorstore...")
    vs = load_vectorstore()

    print("\n" + "=" * 70)
    print("Evaluating Baseline (Similarity Search, k=4)")
    print("=" * 70)
    baseline = evaluate_baseline(vs, QUERIES_AND_GT)
    print(f"  Precision@3: {baseline['Precision@3']}")
    print(f"  Precision@6: {baseline['Precision@6']}")
    print(f"  MRR:         {baseline['MRR']}")

    print("\n" + "=" * 70)
    print("Evaluating Optimized (MMR k=6 + Cross-Encoder Re-ranking top-3)")
    print("=" * 70)
    optimized = evaluate_optimized(vs, QUERIES_AND_GT)
    print(f"  Precision@3: {optimized['Precision@3']}")
    print(f"  Precision@6: {optimized['Precision@6']}")
    print(f"  MRR:         {optimized['MRR']}")

    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"Baseline  → P@3: {baseline['Precision@3']}, P@6: {baseline['Precision@6']}, MRR: {baseline['MRR']}")
    print(f"Optimized → P@3: {optimized['Precision@3']}, P@6: {optimized['Precision@6']}, MRR: {optimized['MRR']}")

    # Calculate improvements
    p3_improvement = optimized["Precision@3"] - baseline["Precision@3"]
    p6_improvement = optimized["Precision@6"] - baseline["Precision@6"]
    mrr_improvement = optimized["MRR"] - baseline["MRR"]

    print("\n" + "=" * 70)
    print("Improvements (Optimized - Baseline)")
    print("=" * 70)
    print(f"P@3: {p3_improvement:+.3f}")
    print(f"P@6: {p6_improvement:+.3f}")
    print(f"MRR: {mrr_improvement:+.3f}")


if __name__ == "__main__":
    main()
