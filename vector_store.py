import faiss
import os
import pickle
import numpy as np
from typing import List

def save_to_faiss(embeddings: List[List[float]], texts: List[str], index_path: str):
    """
    Saves embeddings and corresponding texts to FAISS index and a sidecar pickle file.

    Args:
        embeddings: List of vector embeddings (lists of floats)
        texts: Original text chunks corresponding to the embeddings
        index_path: Path (without extension) to save the FAISS index
    """
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))

    # Save index
    faiss.write_index(index, f"{index_path}.index")

    # Save text mapping
    with open(f"{index_path}.pkl", "wb") as f:
        pickle.dump(texts, f)


def load_faiss(index_path: str):
    """
    Loads FAISS index and the corresponding text mapping.

    Returns:
        index: FAISS index object
        texts: List of corresponding text chunks
    """
    index = faiss.read_index(f"{index_path}.index")

    with open(f"{index_path}.pkl", "rb") as f:
        texts = pickle.load(f)

    return index, texts
