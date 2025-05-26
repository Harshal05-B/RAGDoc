from fastapi import FastAPI
from app import router  # import the router

app = FastAPI(
    title="RAGDoc API",
    description="Query documents using Mistral AI with a clean RAG backend.",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Welcome to RAGDoc API. Visit /docs to interact."}

# Register your routes
app.include_router(router)
