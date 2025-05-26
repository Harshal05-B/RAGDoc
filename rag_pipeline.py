from pdf_loader import load_pdf
from text_splitter import split_text
from embedder import generate_embeddings
from vector_store import save_to_faiss, load_faiss
from llm_interface import get_llm_response

import numpy as np

def ingest_document(pdf_path: str, index_path: str):
    """
    Extracts, chunks, embeds and saves the PDF to FAISS.
    """
    print("[INFO] Loading PDF...")
    text = load_pdf(pdf_path)

    print("[INFO] Splitting text...")
    chunks = split_text(text)
    print(f"[INFO] Split into {len(chunks)} chunks.")

    print("[INFO] Generating embeddings...")
    embeddings = generate_embeddings(chunks)
    print(f"[INFO] Generated {len(embeddings)} embeddings.")

    print("[INFO] Saving embeddings to FAISS...")
    save_to_faiss(embeddings, chunks, index_path)

    print(f"[DONE] Document ingested and stored at {index_path}.index")


def answer_query(index_path: str, query: str, top_k: int = 3) -> str:
    """
    Retrieves top chunks and sends them to Mistral chat.
    """
    print("[INFO] Loading FAISS index...")
    index, texts = load_faiss(index_path)

    print("[INFO] Embedding the query...")
    query_embedding = generate_embeddings([query])[0]

    print("[INFO] Searching for similar chunks...")
    D, I = index.search(np.array([query_embedding]).astype('float32'), top_k)
    retrieved_chunks = [texts[i] for i in I[0]]

    context = "\n\n".join(retrieved_chunks)

    print("[INFO] Generating LLM response...")
    response = get_llm_response(context=context, query=query)
    print("[DONE] Response generated.")
    print("Answer",response)
    return response.strip() or "No relevant information found in the document."
