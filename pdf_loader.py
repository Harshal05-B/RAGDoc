from pathlib import Path
from typing import Union
from PyPDF2 import PdfReader

def load_pdf(file_path: Union[str, Path]) -> str:
    """
    Extracts text content from a PDF file.

    Args:
        file_path (str or Path): Path to the PDF document.

    Returns:
        str: Full text extracted from the PDF.
    """
    file_path = Path(file_path)
    if not file_path.exists() or not file_path.suffix.lower() == ".pdf":
        raise FileNotFoundError(f"Invalid file: {file_path}")

    reader = PdfReader(str(file_path))
    full_text = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text.append(text)

    return "\n".join(full_text)
