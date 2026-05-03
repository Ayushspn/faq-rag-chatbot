import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from PIL import Image
import requests

# --- 1. Load models ---
clip_model = SentenceTransformer("clip-ViT-B-32")
generator = pipeline("text-generation", model="gpt2")

# --- 2. Prepare documents ---
texts = [
    "A diagram of a transformer architecture.",
    "An image showing attention heads in a neural network.",
    "A photo of a cat sitting on a laptop."
]

image_urls = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformer_architecture.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/attention_heads.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat_laptop.jpg"
]
images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

# --- 3. Encode documents ---
text_emb = clip_model.encode(texts, convert_to_tensor=True)
image_emb = clip_model.encode(images, convert_to_tensor=True)

# --- 4. Streamlit UI ---
st.title("🖼️ Multi-Modal RAG Assistant")

if query := st.chat_input("Ask me something..."):
    st.chat_message("user").write(query)

    # Encode query
    query_emb = clip_model.encode(query, convert_to_tensor=True)

    # Retrieve best matches
    best_text_idx = util.cos_sim(query_emb, text_emb).argmax()
    best_image_idx = util.cos_sim(query_emb, image_emb).argmax()

    retrieved_text = texts[best_text_idx]
    retrieved_image = image_urls[best_image_idx]

    # Build prompt for LLM
    prompt = f"""
    Question: {query}
    Retrieved Text: {retrieved_text}
    Retrieved Image: {retrieved_image}

    Answer: Explain the concept using both text and image reference.
    """

    # Generate answer
    answer = generator(prompt, max_length=150, do_sample=True)[0]["generated_text"]

    # Display assistant answer
    st.chat_message("assistant").write(answer)

    # Show retrieved sources
    st.subheader("📚 Sources Used")
    st.write("Text:", retrieved_text)
    st.image(retrieved_image, caption="Retrieved Image")
