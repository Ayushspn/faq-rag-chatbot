from sentence_transformers import SentenceTransformer, util
from PIL import Image
import requests

# --- 1. Load a multi-modal model (CLIP) ---
model = SentenceTransformer("clip-ViT-B-32")

# --- 2. Prepare data ---
# Text documents
texts = [
    "A diagram of a transformer architecture.",
    "An image showing attention heads in a neural network.",
    "A photo of a cat sitting on a laptop."
]

# Image documents
image_urls = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformer_architecture.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/attention_heads.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat_laptop.jpg"
]
images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

# --- 3. Encode both text and images into the same embedding space ---
text_emb = model.encode(texts, convert_to_tensor=True)
image_emb = model.encode(images, convert_to_tensor=True)

# --- 4. Query ---
query = "show me a diagram of transformer attention"
query_emb = model.encode(query, convert_to_tensor=True)

# --- 5. Compute similarity between query and all embeddings ---
text_scores = util.cos_sim(query_emb, text_emb)
image_scores = util.cos_sim(query_emb, image_emb)

# --- 6. Retrieve top match from both modalities ---
best_text_idx = text_scores.argmax()
best_image_idx = image_scores.argmax()

print("Best matching text:", texts[best_text_idx])
print("Best matching image:", image_urls[best_image_idx])
