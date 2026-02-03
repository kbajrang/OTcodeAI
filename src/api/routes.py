from fastapi import APIRouter
from src.models.schema import QueryRequest, QueryResponse
from src.rag.pipeline import GraphRAGPipeline

router = APIRouter()
pipeline = GraphRAGPipeline()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = pipeline.answer(request.question)
    return QueryResponse(answer=result)


@router.get("/")
def root() -> dict:
    return {"status": "ok", "message": "GraphRAG API is running"}


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}
