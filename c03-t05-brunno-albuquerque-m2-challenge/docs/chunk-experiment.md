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

| Configuration | Avg chunks retrieved | Relevant? (1-5) | Answer quality (1-5) | Notes |
|---|---|---|---|---|
| chunk_size=250, overlap=25 | | | | |
| chunk_size=500, overlap=50 (baseline) | | | | |
| chunk_size=1000, overlap=100 | | | | |

## Analysis

<!-- For each configuration, write 1–2 sentences about what you observed. -->

### chunk_size=250, overlap=25
<!-- Example: Smaller chunks cause fragmentation — warranty clause splits across
     two chunks, and the LLM cannot reconstruct the full rule from either alone. -->

### chunk_size=500, overlap=50 (baseline)
<!-- Your observations here. -->

### chunk_size=1000, overlap=100
<!-- Example: Larger chunks return more complete context but retrieved fewer
     unique documents — one long chunk dominates the context window. -->

## Recommendation

<!-- State which configuration you chose for your final pipeline and why.
     Consider the trade-off between context completeness and retrieval diversity.
     Reference your data from the table above to justify the recommendation. -->
