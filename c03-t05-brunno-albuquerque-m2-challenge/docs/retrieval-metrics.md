# Retrieval Quality Metrics — Stop 2

## Instructions

Define a small evaluation set of 10 questions, each with known relevant documents
(by filename). Then measure Precision@k and MRR for both:
- **Baseline**: the simple similarity retriever from Stop 1 (k=4).
- **Optimized**: MMR (k=6, fetch_k=20) + cross-encoder re-ranking (top-3).

Implement the metric functions in your pipeline before filling this table. The
capstone brief provides reference implementations in Stop 2, Component 4.

## Evaluation Set

| # | Question | Relevant documents (filenames) |
|---|---|---|
| 1 | "What is the return window for a refund?" | policy_return_policy.txt |
| 2 | "How do I reset the Router NX300?" | product_manual_router_nx300.txt, support_router_wont_connect.txt |
| 3 | "What does the Premium Protection Plan cover?" | policy_warranty_terms.txt |
| 4 | "Steps to file a warranty claim online" | support_warranty_claim_process.txt |
| 5 | "Laptop Pro X1 specifications" | product_manual_laptop_pro_x1.txt, laptop_specs.csv |
| 6 | "How do I pair a Zigbee device with the Smart Hub?" | product_manual_smart_hub_home.txt |
| 7 | "Does TechStore Plus cover accidental damage?" | policy_warranty_terms.txt |
| 8 | "Laptop won't turn on — first troubleshooting step" | support_laptop_wont_power_on.txt |
| 9 | "What is the restocking fee for an opened product?" | policy_return_policy.txt |
| 10 | "Warranty period for networking equipment" | policy_warranty_terms.txt, product_manual_router_nx300.txt |

## Results

| Pipeline | Precision@3 | Precision@6 | MRR |
|---|---|---|---|
| Baseline (similarity, k=4) | | | |
| Optimized (MMR k=6 + re-rank top-3) | | | |

## Analysis

<!-- Compare the two pipelines. Which queries improved the most after adding
     MMR and re-ranking? Did any queries get worse? Explain why. -->

### Precision@k observations
<!-- Example: MMR improved Precision@6 on query 2 (router reset) because the
     simple retriever returned 4 chunks from the same document, whereas MMR
     forced diversity and surfaced the support article. -->

### MRR observations
<!-- Example: Re-ranking significantly boosted MRR on query 10 — the most
     relevant warranty+networking chunk scored low on cosine similarity but high
     on cross-encoder relevance. -->

## Conclusion

<!-- State whether the optimized pipeline meets the Stop 2 requirement:
     "the optimized pipeline must match or exceed the baseline on MRR." -->
