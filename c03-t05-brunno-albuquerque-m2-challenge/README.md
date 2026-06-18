# TechStore Plus RAG Knowledge Base — M2 Capstone

> Full brief: [../m2-capstone-rag-knowledge-base.md](../m2-capstone-rag-knowledge-base.md)

You are building a production-quality RAG system for TechStore Plus — a technology
e-commerce company. The system grounds every answer in a corpus of product manuals,
support articles, and policy documents. You build it progressively over three weeks:

| Stop | Week | Type | What you build |
|------|------|------|----------------|
| Stop 1 | W4 | Formative | Basic end-to-end RAG pipeline |
| Stop 2 | W5 | Formative | MMR retrieval + cross-encoder re-ranking + metrics |
| Stop 3 | W6 | **Graded** | Graph RAG + guardrails + multimodal + TechStoreRAGAgent |

Only Stop 3 is graded. Rubric: Functionality 40 | Code Quality 25 | Comprehension 20 | Docs 15.

---

## Project structure

```
capstone/
├── .gitignore
├── README.md                           (this file)
├── env.example                         (copy to .env)
├── requirements.txt
├── data/
│   ├── product_manual_laptop_pro_x1.txt
│   ├── product_manual_router_nx300.txt
│   ├── product_manual_smart_hub_home.txt
│   ├── support_router_wont_connect.txt
│   ├── support_warranty_claim_process.txt
│   ├── support_laptop_wont_power_on.txt
│   ├── policy_return_policy.txt
│   ├── policy_warranty_terms.txt
│   └── tables/
│       ├── laptop_specs.csv            (5 models, RAM/storage/price/tier)
│       └── warranty_tiers.csv          (basic/standard/premium)
├── docs/
│   ├── chunk-experiment.md             (fill in during Stop 2)
│   └── retrieval-metrics.md            (fill in during Stop 2)
├── src/
│   ├── __init__.py
│   ├── rag_agent.py                    (TechStoreRAGAgent + GuardrailedAnswer)
│   ├── pipeline/
│   │   ├── loader.py                   (Stop 1)
│   │   ├── vectorstore.py              (Stop 1 + Stop 2)
│   │   └── reranker.py                 (Stop 2)
│   ├── graph/
│   │   └── knowledge_graph.py          (Stop 3)
│   ├── guardrails/
│   │   ├── writer.py                   (Stop 3)
│   │   └── verifier.py                 (Stop 3)
│   └── multimodal/
│       └── table_retriever.py          (Stop 3)
└── tests/
    └── test_mandatory_cases.py
```

---

## Setup

### 1. Copy this starter kit

```bash
cp -R weekly-content/modules/02-rag-systems/capstone/ my-rag-kb/
cd my-rag-kb/
git init && git add -A && git commit -m "starter: m2 capstone rag knowledge base"
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp env.example .env
# Edit .env — add your OPENAI_API_KEY
```

### 5. Verify Python version

```bash
python --version   # must be 3.11+
```

---

## Running the tests

```bash
# Run the GuardrailedAnswer smoke test (works before any implementation)
pytest tests/test_mandatory_cases.py::test_guardrailed_answer_valid_construction -v

# Run all mandatory cases (skipped until you implement TechStoreRAGAgent)
pytest tests/test_mandatory_cases.py -v

# Remove @pytest.mark.skip from each test when you are ready to run it
```

---

## What to implement at each stop

### Stop 1 — Week 4: Basic RAG pipeline

| File | What to implement |
|------|-------------------|
| `src/pipeline/loader.py` | `load_documents()`, `chunk_documents()` |
| `src/pipeline/vectorstore.py` | `build_vectorstore()`, `load_vectorstore()` |

After Stop 1, you should be able to run a simple RAG chain from a script:

```python
from src.pipeline.loader import load_documents, chunk_documents
from src.pipeline.vectorstore import build_vectorstore, load_vectorstore

docs   = load_documents()
chunks = chunk_documents(docs)
vs     = build_vectorstore(chunks)   # first run — calls OpenAI embeddings API

# On subsequent runs, reload from disk:
vs = load_vectorstore()
results = vs.similarity_search("return policy", k=4)
```

Tag your work: `git tag stop-1 && git push origin main --tags`

---

### Stop 2 — Week 5: Optimized retrieval

| File | What to implement |
|------|-------------------|
| `src/pipeline/vectorstore.py` | `get_mmr_retriever()` |
| `src/pipeline/reranker.py` | `rerank()` |
| `docs/chunk-experiment.md` | Fill in the table with your experiment results |
| `docs/retrieval-metrics.md` | Fill in Precision@k and MRR comparison |

After Stop 2:

```python
from src.pipeline.vectorstore import load_vectorstore, get_mmr_retriever
from src.pipeline.reranker import rerank

vs       = load_vectorstore()
retriever = get_mmr_retriever(vs)
mmr_docs  = retriever.invoke("differences between standard and extended warranty")
top_docs  = rerank("differences between standard and extended warranty", mmr_docs)
```

Tag: `git tag stop-2 && git push origin main --tags`

---

### Stop 3 — Week 6: Production system (graded)

| File | What to implement |
|------|-------------------|
| `src/graph/knowledge_graph.py` | `TechStoreKnowledgeGraph` (all methods) |
| `src/guardrails/writer.py` | `build_cited_answer()` |
| `src/guardrails/verifier.py` | `verify_answer()` |
| `src/multimodal/table_retriever.py` | `TableRetriever` (all methods) |
| `src/rag_agent.py` | `TechStoreRAGAgent._ensure_initialized()`, `answer()` |

After Stop 3, the three mandatory test cases must pass:

```bash
# Remove @pytest.mark.skip from each test first, then:
pytest tests/test_mandatory_cases.py -v
```

Entry point usage:

```python
from src.rag_agent import TechStoreRAGAgent

agent = TechStoreRAGAgent()

# On-topic query
r = agent.answer("What is the return period for a refund?")
print(r.answer)          # cited answer string
print(r.decision)        # "answer" | "answer_with_disclaimer" | "extractive" | "no_answer"
print(r.cited_sources)   # list of source identifiers

# Off-topic query — decision gate fires
r2 = agent.answer("What is the capital of France?")
assert r2.decision == "no_answer"
```

M1 integration:

```python
from src.rag_agent import TechStoreRAGAgent
from langchain_core.tools import tool

rag = TechStoreRAGAgent()

@tool
def search_knowledge_base(question: str) -> str:
    """Search TechStore Plus product manuals, support articles, and policies."""
    result = rag.answer(question)
    return result.answer
```

Tag and submit: `git tag stop-3 && git push origin main --tags`
Then paste your repo URL in the Moodle M2 Capstone submission field.

---

## Component interaction diagram

```
User question
     │
     ▼
TechStoreRAGAgent.answer()
     │
     ├──[always]────────────────────────────────────────────────────────┐
     │   pipeline/vectorstore.get_mmr_retriever()                       │
     │        └─→ MMR (fetch_k=20, k=6) → pipeline/reranker.rerank()  │
     │                                     (top-3 docs)                 │
     │                                                                   │
     ├──[entity-dense query]────────────────────────────────────────────┤
     │   graph/knowledge_graph.query_subgraph()                         │
     │        └─→ BFS traversal (≤2 hops, relation allowlist)           │
     │                                                                   │
     ├──[numeric/table query]───────────────────────────────────────────┤
     │   multimodal/table_retriever.retrieve()                          │
     │        └─→ CSV row matching + superlative sort                   │
     │                                                                   │
     └──[merge all evidence]────────────────────────────────────────────┘
          │
          ▼
     guardrails/writer.build_cited_answer()
          └─→ every sentence ends with [source_key]
               │
               ▼
     guardrails/verifier.verify_answer()
          └─→ claim decomposition → entailment → decision gate
               │
               ▼
          GuardrailedAnswer
```

---

## Known limitations

- The knowledge graph is in-memory only — it is rebuilt on every cold start.
  Persistence (serialise to JSON) is a recommended extension.
- Triple extraction quality depends on GPT-4.1-mini prompt quality; some
  low-confidence triples may be included. Use DEFAULT_RELATION_ALLOWLIST to
  restrict noise.
- TableRetriever does not use embeddings by default (Stop 3 baseline uses
  keyword matching). For higher recall, implement Option B (embedding-based
  row retrieval) as described in the module docstring.
- The ChromaDB collection must be rebuilt if you change CHUNK_SIZE or the
  embedding model. Delete `chroma_db/` and re-run `build_vectorstore()`.
