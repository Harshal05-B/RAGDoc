from pdf_loader import load_pdf
from text_splitter import split_text
from embedder import get_embedding_model, generate_embeddings
from vector_store import save_to_faiss, load_faiss
from llm_interface import get_llm_response 

import numpy as np
from typing import List

def ingest_document(pdf_path: str, index_path: str):
    """
    Complete pipeline to process a PDF and store its embeddings.
    """
    print("[INFO] Loading PDF...")
    text = load_pdf(pdf_path)

    print("[INFO] Splitting text...")
    chunks = split_text(text)

    print("[INFO] Generating embeddings...")
    tokenizer, model = get_embedding_model()
    embeddings = generate_embeddings(tokenizer, model, chunks)

    print("[INFO] Saving embeddings to FAISS...")
    save_to_faiss(embeddings, chunks, index_path)

    print(f"[DONE] Document ingested and stored at {index_path}.index")


def answer_query(index_path: str, query: str, top_k: int = 3) -> str:
    """
    Retrieves relevant chunks and queries the LLM.
    """
    print("[INFO] Loading FAISS index...")
    index, texts = load_faiss(index_path)

    print("[INFO] Embedding the query...")
    tokenizer, model = get_embedding_model()
    query_embedding = generate_embeddings(tokenizer, model, [query])[0]

    print("[INFO] Searching for similar chunks...")
    D, I = index.search(np.array([query_embedding]).astype('float32'), top_k)
    retrieved_chunks = [texts[i] for i in I[0]]

    context = "\n\n".join(retrieved_chunks)

    print("[INFO] Generating LLM response...")
    response = get_llm_response(context=context, query=query)

    return response
