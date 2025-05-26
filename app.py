from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
from rag_pipeline import ingest_document, answer_query

app = FastAPI(
    title="RAGDoc API",
    description="API for querying documents using Mistral AI + RAG.",
    version="1.0.0"
)

DATA_DIR = Path("data")
VECTOR_DIR = Path("vector_store")
DATA_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)


@app.post("/upload", summary="Upload and index a PDF document")
async def upload_document(file: UploadFile = File(...)):
    file_path = DATA_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    index_path = VECTOR_DIR / file.filename.replace(".pdf", "")
    ingest_document(str(file_path), str(index_path))

    return JSONResponse({"message": "Document indexed successfully."})


@app.post("/query", summary="Ask a question about a previously indexed PDF")
async def query_document(query: str = Form(...), file_name: str = Form(...)):
    index_path = VECTOR_DIR / file_name.replace(".pdf", "")
    
    if not (index_path.with_suffix(".index")).exists():
        return JSONResponse({"error": "Index not found for the specified file."}, status_code=404)

    response = answer_query(str(index_path), query)
    return JSONResponse({"response": response})
