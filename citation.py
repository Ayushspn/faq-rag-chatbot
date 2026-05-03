import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

# Embedding + DB setup
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./citation_db")
collection = client.get_or_create_collection("citation_collection")

generator = pipeline("text-generation", model="gpt2")

st.title("📚 Citation-Aware RAG Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if query := st.chat_input("Ask me something..."):
    st.session_state.history.append({"role":"user","content":query})
    st.chat_message("user").write(query)

    # Build expanded query with history
    history_text = " ".join([m["content"] for m in st.session_state.history])
    query_emb = embedder.encode([history_text]).tolist()

    # Retrieve context
    results = collection.query(query_embeddings=query_emb, n_results=3)
    retrieved_chunks = results["documents"][0]

    # Label chunks with IDs
    labeled_context = "\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks)])

    # Prompt with citation instruction
    prompt = f"""
Conversation: {history_text}
Context (with sources):
{labeled_context}

Question: {query}
Answer using the sources and cite them (e.g., [1], [2]).
"""
    result = generator(prompt, max_length=200, do_sample=True)
    answer = result[0]["generated_text"]

    # Save + display
    st.session_state.history.append({"role":"assistant","content":answer})
    st.chat_message("assistant").write(answer)

    # Show sources separately
    st.subheader("Sources Used")
    for i, chunk in enumerate(retrieved_chunks):
        st.write(f"[{i+1}] {chunk}")
