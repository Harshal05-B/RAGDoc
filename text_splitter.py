from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Splits input text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.

    Args:
        text (str): Full document text.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Number of characters to overlap between chunks.

    Returns:
        list[str]: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(text)
    return chunks
