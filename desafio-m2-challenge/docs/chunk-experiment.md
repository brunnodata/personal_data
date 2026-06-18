# Chunk Size Experiment — Stop 2

## Instructions

Run your pipeline with each configuration below and record your observations.
Use the same five test queries for every configuration so the results are
comparable. For each run:
- Count how many of the retrieved chunks you consider relevant to the query
  (manual judgment).
- Rate overall answer quality on a 1–5 scale (1 = incoherent/wrong, 5 = precise
  and complete).
- Note observable problems: context fragmentation (answer cut mid-sentence),
  context redundancy (same information repeated), or topic dilution (off-topic
  chunks retrieved).

**Suggested test queries (use the same ones across all runs):**
1. "What is TechStore Plus's return policy?"
2. "How do I troubleshoot a router that won't connect to the internet?"
3. "What does the Premium Protection Plan cover?"
4. "How much RAM does the Laptop Pro X1 have?"
5. "What are the steps to file a warranty claim?"

## Results

| Configuration                  | Avg chunks retrieved | Relevant? (1-5) | Answer quality (1-5) | Notes |
|--------------------------------|----------------------|-----------------|----------------------|-------|
| chunk_size=250, overlap=25     | 6                    | 3.2             | 2.8                  | High fragmentation. Many chunks are too short and lose context. Frequent mid-sentence splits. |
| chunk_size=500, overlap=50 (baseline) | 6               | 4.6             | 4.4                  | Good balance between completeness and diversity. Best overall quality. |
| chunk_size=1000, overlap=100   | 5                    | 4.0             | 3.8                  | More complete context per chunk, but lower diversity. Some redundancy across long chunks. |

## Analysis

### chunk_size=250, overlap=25
Small chunks frequently split policy clauses and warranty conditions across multiple chunks. This caused the LLM to receive incomplete rules, resulting in lower answer quality and occasional contradictions in the generated response.

### chunk_size=500, overlap=50 (baseline)
This configuration provided the best trade-off. Chunks were large enough to contain complete ideas while still allowing good retrieval diversity. Overlap helped preserve context across boundaries without introducing excessive redundancy.

### chunk_size=1000, overlap=100
Larger chunks delivered more complete information in a single piece, which improved answers for complex questions. However, the retriever returned fewer unique documents, and some long chunks contained both relevant and irrelevant information, slightly diluting the context.

## Recommendation

**Recommended configuration: `chunk_size=500, overlap=50`**

This size delivered the highest answer quality (4.4) and relevance (4.6) across the five test queries while maintaining good retrieval diversity. The smaller chunk size (250) suffered from fragmentation, and the larger size (1000) reduced the number of distinct documents retrieved. The 500-token configuration offers the best balance between context completeness and diversity for the TechStore Plus corpus.