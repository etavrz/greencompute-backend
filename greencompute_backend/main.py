from fastapi import FastAPI
from loguru import logger

import greencompute_backend.routes as routers

from .config import ENVIRON, ROOT_PATH
from .db.engine import engine
from .db.tables import Base

logger.debug(f"Running on {ENVIRON} env with root path {ROOT_PATH}")
app = FastAPI(root_path=ROOT_PATH)

# Table creation
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    logger.error(f"Could not create tables: {e}")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


app.include_router(routers.llm_router)
app.include_router(routers.retrieval_router)
app.include_router(routers.models_router)
