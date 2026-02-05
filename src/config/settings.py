from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.paths import PROJECT_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Models
    # Providers: "ollama" (recommended for local), or "openai_compat"
    llm_provider: str = "ollama"
    # For Ollama, set to "http://localhost:11434"
    # For OpenAI-compatible servers, set to e.g. "http://localhost:8001/v1"
    llama_api_base: str = "http://localhost:11434"
    llama_api_key: str = ""
    # Ollama example: "codellama:13b" (or your pulled tag)
    llama_model_name: str = "codellama:13b"
    # Keep this low so UI errors fast if Ollama isn't running.
    llama_timeout: int = 0
    embedding_model: str = "BAAI/bge-base-en"
    embedding_batch_size: int = 32
    # When sentence-transformers is unavailable, we fall back to hashing embeddings.
    # Keep this at 768 by default to match common BGE dimensions and avoid FAISS dim mismatches.
    fallback_embedding_dim: int = 768
    llm_num_ctx: int = 4096
    prompt_reserved_tokens: int = 512
    approx_chars_per_token: int = 4

    # Storage
    index_dir: str = "data/index"
    graph_path: str = "data/graph/graph.pkl"
    vector_db_path: str = "data/vector/faiss.index"
    vector_metadata_path: str = "data/vector/metadata.json"
    modules_dir: str = "modules"

    # Parsing
    languages: list[str] = ["python", "javascript", "typescript", "java"]
    chunk_size_lines: int = 120
    chunk_overlap_lines: int = 20

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
