from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi

# Sample documents
docs = [
    "Python was created by Guido van Rossum in 1991.",
    "Guido van Rossum was born in 1956 in the Netherlands.",
    "Transformers use attention mechanisms for NLP tasks."
]

# --- 1. Setup Embedding Search (Semantic) ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection("hybrid")

# Add docs to ChromaDB
embeddings = embedder.encode(docs).tolist()
collection.add(documents=docs, embeddings=embeddings, ids=[str(i) for i in range(len(docs))])

# --- 2. Setup Keyword Search (BM25) ---
tokenized_docs = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)

# --- 3. Hybrid Search Function ---
def hybrid_search(query, top_k=2):
    # Semantic search
    query_emb = embedder.encode([query]).tolist()
    semantic_results = collection.query(query_embeddings=query_emb, n_results=top_k)["documents"][0]

    # Keyword search
    keyword_results = bm25.get_top_n(query.split(), docs, n=top_k)

    # Merge (simple union for demo)
    combined = list(set(semantic_results + keyword_results))
    return combined

# --- 4. Example Query ---
query = "When was Guido born?"
results = hybrid_search(query, top_k=2)

print("Hybrid Search Results:")
for r in results:
    print("-", r)
