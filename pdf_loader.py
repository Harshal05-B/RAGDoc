import fitz  # PyMuPDF
from pathlib import Path
from typing import Union

def load_pdf(file_path: Union[str, Path]) -> str:
    """
    Extracts text from each page of a PDF using PyMuPDF (fitz).

    Args:
        file_path (str or Path): Path to the PDF file.

    Returns:
        str: Combined text from all pages.
    """
    file_path = Path(file_path)
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"Invalid PDF file: {file_path}")

    doc = fitz.open(str(file_path))
    text = ""
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        print(f"[INFO] Page {page_num}: {len(page_text)} characters extracted.")
        text += page_text + "\n"

    doc.close()
    print(f"[INFO] Total text extracted: {len(text)} characters.")
    if not text.strip():
        raise ValueError("No text found in the PDF document.")
    print("[DONE] PDF text extraction completed.")
    print("text is",text[:500])  # Print first 500 characters for debugging
    return text.strip()
