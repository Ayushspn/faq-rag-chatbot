import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# --- 1. Load model ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text-generation", model="distilgpt2")

# --- 2. FAQ dataset ---
docs = [
    "Q: How do I reset my password? A: Go to settings and click 'Reset Password'.",
    "Q: What is the refund policy? A: Refunds are available within 30 days of purchase.",
    "Q: How can I contact support? A: Email support@startup.com."
]

# --- 3. Build FAISS index ---
embeddings = embedder.encode(docs)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# --- 4. Streamlit UI ---
st.title("💡 Startup FAQ Assistant")

query = st.text_input("Ask a question:")

if query:
    query_emb = embedder.encode([query])
    D, I = index.search(query_emb, k=2)
    retrieved = [docs[i] for i in I[0]]

    # Build prompt with citations
    context = "\n".join([f"[{j+1}] {doc}" for j, doc in enumerate(retrieved)])
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer (cite sources):"

    answer = generator(prompt, max_length=100, do_sample=True)[0]["generated_text"]

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for j, doc in enumerate(retrieved):
        st.write(f"[{j+1}] {doc}")
