"""
Shield-Flow: embed.py
=====================
Converts transactions into text embeddings and stores them in Pinecone.
Used by the API to find similar historical fraud cases for each new transaction.

Usage:
  python services/model/embed.py
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = "fraud-cases"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 outputs 384-dim vectors
BATCH_SIZE = 100  # upsert 100 vectors at a time
MAX_ROWS = 10000  # free tier: 1M vectors max, 10k is plenty to start

# ─────────────────────────────────────────────
# Step 1: Connect to Pinecone
# ─────────────────────────────────────────────
print("\n[1/4] Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist yet
existing = [i.name for i in pc.list_indexes()]
if INDEX_NAME not in existing:
    print(f"  Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("  Index created.")
else:
    print(f"  Index '{INDEX_NAME}' already exists.")

index = pc.Index(INDEX_NAME)
print(f"  Connected. Stats: {index.describe_index_stats()}")

# ─────────────────────────────────────────────
# Step 2: Load transactions
# ─────────────────────────────────────────────
print(f"\n[2/4] Loading {MAX_ROWS:,} transactions...")
df = pd.read_csv("data/raw/train_transaction.csv", nrows=MAX_ROWS)
print(
    f"  Loaded: {len(df):,} rows | Fraud: {df['isFraud'].sum():,} ({df['isFraud'].mean():.2%})"
)

# ─────────────────────────────────────────────
# Step 3: Convert transactions to text + embed
# ─────────────────────────────────────────────
print("\n[3/4] Generating embeddings...")


def tx_to_text(row):
    """
    Turn a transaction row into a natural language sentence.
    This is what gets embedded — the richer the text, the better the similarity search.
    """
    amount = f"${row['TransactionAmt']:.2f}"
    domain = (
        row.get("P_emaildomain", "unknown")
        if pd.notna(row.get("P_emaildomain"))
        else "unknown"
    )
    card = row.get("card4", "unknown") if pd.notna(row.get("card4")) else "unknown"
    product = (
        row.get("ProductCD", "unknown") if pd.notna(row.get("ProductCD")) else "unknown"
    )
    addr = row.get("addr1", "unknown") if pd.notna(row.get("addr1")) else "unknown"
    fraud = "FRAUD" if row["isFraud"] == 1 else "LEGIT"

    return (
        f"{fraud} transaction of {amount} "
        f"using {card} card "
        f"product type {product} "
        f"email domain {domain} "
        f"billing region {addr}"
    )


# Convert all rows to text
texts = df.apply(tx_to_text, axis=1).tolist()
print(f"  Example text: '{texts[0]}'")
print(f"  Example fraud text: '{texts[df['isFraud'].idxmax()]}'")

# Load the embedding model (downloads ~90MB on first run)
print("  Loading sentence-transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)
print(f"  Embeddings shape: {embeddings.shape}")  # (10000, 384)

# ─────────────────────────────────────────────
# Step 4: Upsert to Pinecone
# ─────────────────────────────────────────────
print("\n[4/4] Upserting vectors to Pinecone...")

vectors = [
    (
        str(i),  # unique ID
        embeddings[i].tolist(),  # 384-dim vector
        {  # metadata — stored alongside vector
            "isFraud": int(df.iloc[i]["isFraud"]),
            "amount": float(df.iloc[i]["TransactionAmt"]),
            "email_domain": str(df.iloc[i].get("P_emaildomain", "unknown")),
            "card_type": str(df.iloc[i].get("card4", "unknown")),
            "product": str(df.iloc[i].get("ProductCD", "unknown")),
            "text": texts[i],
        },
    )
    for i in range(len(embeddings))
]

# Upsert in batches of 100 (Pinecone rate limit)
total_upserted = 0
for start in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[start : start + BATCH_SIZE]
    index.upsert(vectors=batch)
    total_upserted += len(batch)
    if total_upserted % 1000 == 0:
        print(f"  Upserted {total_upserted:,} / {len(vectors):,}")

print(f"\n  Done — {total_upserted:,} vectors in Pinecone.")
print(f"  Final index stats: {index.describe_index_stats()}")
print("\nPhase 3 complete. Pinecone is ready for similarity search.")
