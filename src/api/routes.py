from fastapi import APIRouter
from src.models.schema import QueryRequest, QueryResponse
from src.rag.pipeline import GraphRAGPipeline

router = APIRouter()
pipeline = GraphRAGPipeline()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = pipeline.answer(request.question)
    return QueryResponse(answer=result)
