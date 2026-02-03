from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Models
    llm_model: str = "codellama:13b"
    embedding_model: str = "BAAI/bge-base-en"

    # Storage
    index_dir: str = "data/index"
    graph_path: str = "data/graph/graph.pkl"
    vector_db_path: str = "data/vector/faiss.index"

    # Parsing
    languages: list[str] = ["python", "javascript", "typescript"]

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
