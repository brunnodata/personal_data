"""
inspect_chroma.py — Script to visually inspect the internal tables of your ChromaDB collection.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from src.pipeline.vectorstore import load_vectorstore

def inspect_database():
    print("🔍 [inspector] Connecting to local ChromaDB...")
    vs = load_vectorstore()
    
    # Extract the underlying raw collection data
    collection_data = vs._collection.get(
        include=["documents", "metadatas", "embeddings"]
    )
    
    documents = collection_data.get("documents", [])
    metadatas = collection_data.get("metadatas", [])
    embeddings = collection_data.get("embeddings", [])
    
    print(f"\n📊 Total records found in table: {len(documents)}")
    
    # Display the first 2 chunks as a visual sample
    sample_limit = min(2, len(documents))
    print(f"\n--- Visualizing Sample Records (Top {sample_limit}) ---")
    
    for i in range(sample_limit):
        print(f"\n[RECORD #{i+1}]")
        print(f"  📂 Source File : {metadatas[i].get('source')}")
        print(f"  📝 Text Snippet : {documents[i][:120]}...")
        # Show just the first 5 dimensions of the 1536 vector matrix
        vector_sample = [round(num, 4) for num in embeddings[i][:5]]
        print(f"  🔢 Embedding Matrix (First 5 dimensions): {vector_sample}...")

if __name__ == "__main__":
    inspect_database()
