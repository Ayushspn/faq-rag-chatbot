import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

# Step 1: Load FAQ file dynamically
with open("faq.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Step 2: Chunking
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


docs = chunk_text(text)

# Step 3: Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: ChromaDB Setup
client = chromadb.PersistentClient(path="./faq_db")
collection = client.get_or_create_collection("faq_collection")

# Add chunks
for i, doc in enumerate(docs):
    emb = embedder.encode([doc]).tolist()
    collection.add(documents=[doc], embeddings=emb, ids=[str(i)])

# Step 5: Streamlit UI
st.title("📚 FAQ Chatbot (RAG)")
query = st.text_input("Ask a question:")

if query:
    query_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=3)
    retrieved_chunks = results["documents"][0]
    context = " ".join(retrieved_chunks)

    generator = pipeline("text-generation", model="gpt2")
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_length=150, do_sample=True)

    st.subheader("Retrieved Context")
    st.write(context)

    st.subheader("Generated Answer")
    st.write(result[0]['generated_text'])
