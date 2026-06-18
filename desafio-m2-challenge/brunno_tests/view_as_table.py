import sys
from pathlib import Path
import pandas as pd
from tabulate import tabulate
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).resolve().parent.parent)) # Ajuste do path para achar a pasta src
from src.pipeline.vectorstore import load_vectorstore

def display_chroma_table() -> None:
    """Fetch all vector records and display them as a formatted terminal table."""
    print("🔍 [table_viewer] Reading collections from local storage...")
    vectorstore = load_vectorstore()

    # 1. ADICIONADO: "embeddings" incluído no parâmetro include
    collection_data = vectorstore._collection.get(
        include=["documents", "metadatas", "embeddings"]
    )

    ids = collection_data.get("ids", [])
    documents = collection_data.get("documents", [])
    metadatas = collection_data.get("metadatas", [])
    embeddings = collection_data.get("embeddings", []) # 2. ADICIONADO: Extração dos vetores

    if not ids:
        print("⚠️ [table_viewer] No records found inside the collection.")
        return

    # Restructure into a flat list of dictionaries for Pandas ingestion
    table_rows = []
    for idx, doc_id in enumerate(ids):
        # Truncate text block to 65 characters to preserve table alignment in terminal
        truncated_text = (
            f"{documents[idx][:65]}..."
            if len(documents[idx]) > 65
            else documents[idx]
        )
        # Standardize missing line endings
        truncated_text = truncated_text.replace("\n", " ")

        if embeddings is not None and idx < len(embeddings) and embeddings[idx] is not None:
            vec = embeddings[idx]
            dim = len(vec)
            # Mostra os 3 primeiros números e a dimensão total ex: [0.12, -0.04, 0.51...] (dim: 1536)
            vec_snippet = f"[{vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}...] (dim: {dim})"
        else:
            vec_snippet = "N/A"

        table_rows.append(
            {
                "Row ID": doc_id,
                "Source File": metadatas[idx].get("source", "N/A"),
                "Vector Embedding": vec_snippet, # 4. ADICIONADO: Nova coluna na tabela
                "Text Chunk Content Snippet": truncated_text,
            }
        )

    # Convert to standard Pandas DataFrame
    df = pd.DataFrame(table_rows)

    # Sort records by filename alphabetically to group matching sources together
    df = df.sort_values(by="Source File").reset_index(drop=True)

    print(f"\n📊 Total Matched Database Rows: {len(df)}")
    print("\n--- ChromaDB Collection Table View (Top 15 Chunks Sample) ---")
    
    # Render table with grid layout using tabulate
    print(tabulate(df.head(15), headers="keys", tablefmt="grid", showindex=True))


if __name__ == "__main__":
    display_chroma_table()
