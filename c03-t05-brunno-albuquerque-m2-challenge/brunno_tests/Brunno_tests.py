import sys
from pathlib import Path

# Aponta para a raiz do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.loader import load_documents, chunk_documents

print("=" * 50)
print("TESTE 1: load_documents()")
print("=" * 50)

docs = load_documents()

print("\n--- Metadados por arquivo ---")
for doc in docs:
    print(f"  source:   {doc.metadata['source']}")
    print(f"  category: {doc.metadata['category']}")
    print(f"  chars:    {len(doc.page_content)}")
    print()

print("=" * 50)
print("TESTE 2: chunk_documents()")
print("=" * 50)

chunks = chunk_documents(docs)

print("\n--- Primeiros 3 chunks ---")
for chunk in chunks[:3]:
    print(f"  source:      {chunk.metadata['source']}")
    print(f"  category:    {chunk.metadata['category']}")
    print(f"  chunk_index: {chunk.metadata['chunk_index']}")
    print(f"  preview:     {chunk.page_content[:80]!r}")
    print()
