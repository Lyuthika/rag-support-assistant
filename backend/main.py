from fastapi import FastAPI
from pydantic import BaseModel

from rag_pipeline import answer_question


app = FastAPI(
    title="RAG Support Assistant",
    description="Backend API for the Notion RAG support assistant",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    title: str
    source_url: str
    similarity: float
    content: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


@app.get("/")
def root():
    return {
        "message": "RAG Support Assistant API is running"
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    result = answer_question(
        question=request.question,
        top_k=5,
    )

    return result