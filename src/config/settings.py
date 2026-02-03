from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Models
    llm_model: str = "codellama:13b-instruct"
    embedding_model: str = "BAAI/bge-base-en"
    ollama_base_url: str = "http://localhost:11434"
    ollama_connect_timeout_s: int = 10
    ollama_read_timeout_s: int = 600
    ollama_stream: bool = True
    ollama_num_ctx: int = 4096
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
