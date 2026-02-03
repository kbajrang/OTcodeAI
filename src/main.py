from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="GraphRAG Code Intelligence")
app.include_router(router)
