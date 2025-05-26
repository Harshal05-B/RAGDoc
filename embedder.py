from typing import List
from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and small, can change

def generate_embeddings(texts: List[str], batch_size: int = 10) -> List[List[float]]:
    """
    Generate embeddings using a local SentenceTransformer model.
    """
    print(f"[INFO] Generating {len(texts)} embeddings using SentenceTransformer.")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings.tolist()
