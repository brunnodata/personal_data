# test_stop_1.py (Crie este arquivo na raiz para testar)
import os
from dotenv import load_dotenv
from src.pipeline.loader import load_documents, chunk_documents
from src.pipeline.vectorstore import build_vectorstore, load_vectorstore

# 1. Load environment variables (OPENAI_API_KEY)
load_dotenv()

print("--- Running Stop 1 Validation Flow ---")

# 2. Execute Ingestion Pipeline
docs = load_documents()
chunks = chunk_documents(docs)

# 3. Build and persist to disk (First run)
print("\n[Step 1] Building vectorstore...")
vs = build_vectorstore(chunks)

# 4. Load from disk (Subsequent runs simulation)
print("\n[Step 2] Loading vectorstore from disk...")
vs_loaded = load_vectorstore()

# 5. Execute a simple similarity search to guarantee it works
print("\n[Step 3] Testing similarity search...")
results = vs_loaded.similarity_search("return policy", k=4)

print(f"\nSuccessfully retrieved {len(results)} chunks from local ChromaDB!")
for i, doc in enumerate(results, 1):
    print(f"  Chunk {i} source: {doc.metadata.get('source')}")
