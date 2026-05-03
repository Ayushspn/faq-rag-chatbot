import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

# Embedding + DB setup
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./conv_db")
collection = client.get_or_create_collection("conv_collection")

generator = pipeline("text-generation", model="gpt2")

st.title("🗨️ Multi-Turn Conversational RAG")

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
    context = " ".join(results["documents"][0])

    # Generate answer
    prompt = f"Conversation: {history_text}\nContext: {context}\nAnswer:"
    result = generator(prompt, max_length=150, do_sample=True)
    answer = result[0]["generated_text"]

    st.session_state.history.append({"role":"assistant","content":answer})
    st.chat_message("assistant").write(answer)
