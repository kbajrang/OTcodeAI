from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Models
    llm_model: str = "codellama:13b"
    embedding_model: str = "BAAI/bge-base-en"
    ollama_base_url: str = "http://localhost:11434"

    # Storage
    index_dir: str = "data/index"
    graph_path: str = "data/graph/graph.pkl"
    vector_db_path: str = "data/vector/faiss.index"
    vector_metadata_path: str = "data/vector/metadata.json"

    # Parsing
    languages: list[str] = ["python", "javascript", "typescript"]

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
