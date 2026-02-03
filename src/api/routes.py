from fastapi import APIRouter, HTTPException

from src.indexer.job import index_job_manager
from src.indexer.workspace import import_module, list_modules
from src.models.schema import (
    IndexRequest,
    IndexStatusResponse,
    ModuleImportRequest,
    ModuleInfoResponse,
    QueryRequest,
    QueryResponse,
)
from src.rag.pipeline import GraphRAGPipeline

router = APIRouter()
pipeline = GraphRAGPipeline()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    # Debug + context-window support is implemented in the pipeline; default response stays compatible.
    result = pipeline.answer(request.question, k=request.k, debug=request.debug)
    if isinstance(result, dict):
        return QueryResponse(**result)
    return QueryResponse(answer=result)


@router.post("/index", response_model=IndexStatusResponse)
def start_index(request: IndexRequest) -> IndexStatusResponse:
    state = index_job_manager.start(
        request.repo_path,
        workspace=request.workspace,
        module_name=request.module_name,
        reset_logs=request.reset_logs,
    )
    return IndexStatusResponse(**state.__dict__)


@router.get("/index/status", response_model=IndexStatusResponse)
def index_status() -> IndexStatusResponse:
    state = index_job_manager.get_state()
    return IndexStatusResponse(**state.__dict__)


@router.get("/modules", response_model=list[ModuleInfoResponse])
def modules() -> list[ModuleInfoResponse]:
    return [ModuleInfoResponse(name=m.name, path=m.path) for m in list_modules()]


@router.post("/modules/import", response_model=ModuleInfoResponse)
def import_module_route(request: ModuleImportRequest) -> ModuleInfoResponse:
    try:
        module = import_module(
            request.source_path,
            request.module_name,
            overwrite=request.overwrite,
        )
        return ModuleInfoResponse(name=module.name, path=module.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/")
def root() -> dict:
    return {"status": "ok", "message": "GraphRAG API is running"}


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}
