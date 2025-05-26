from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def get_embedding_model(model_name: str = "jinaai/jina-embeddings-v2-base-en"):
    """
    Loads the Jina embedding model and tokenizer from Hugging Face.

    Returns:
        tokenizer, model: Hugging Face tokenizer and model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer, model


def mean_pooling(model_output, attention_mask):
    """
    Applies mean pooling to the model output using attention mask.

    Returns:
        torch.Tensor: Pooled sentence embeddings
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def generate_embeddings(tokenizer, model, texts: List[str]) -> List[List[float]]:
    """
    Generates normalized embedding vectors for a list of text inputs.

    Args:
        tokenizer: Hugging Face tokenizer
        model: Hugging Face model
        texts: List of text chunks

    Returns:
        List of float vectors
    """
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

    return normalized_embeddings.tolist()
