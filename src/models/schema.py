from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    k: int = Field(default=5, ge=1, le=25)
    debug: bool = False


class RetrievedContextItem(BaseModel):
    id: str
    score: float | None = None
    type: str | None = None
    name: str | None = None
    module: str | None = None
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    chars: int | None = None


class QueryResponse(BaseModel):
    answer: str
    retrieved: list[RetrievedContextItem] = Field(default_factory=list)
    context: str | None = None
    prompt: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    repo_path: str
    workspace: bool = False
    module_name: str | None = None
    reset_logs: bool = True


class IndexStatusResponse(BaseModel):
    status: str
    stage: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    message: str | None = None
    logs: list[str] = Field(default_factory=list)
    result: dict[str, Any] | None = None


class ModuleInfoResponse(BaseModel):
    name: str
    path: str


class ModuleImportRequest(BaseModel):
    source_path: str
    module_name: str
    overwrite: bool = False
