import datetime

import boto3
from fastapi import Depends, FastAPI
from loguru import logger
from sqlalchemy.orm import Session

from .config import ENVIRON, ROOT_PATH
from .db.engine import engine
from .db.tables import Base, Document
from .resources import get_db, get_s3_client
from .routes import llm_router, models_router

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


@app.get("/buckets")
async def list_buckets(client: boto3.client = Depends(get_s3_client)):
    response = client.list_buckets()
    return response


@app.get("/documents")
def create_document(db: Session = Depends(get_db)):
    document = Document(
        embeddings=[0.1 for _ in range(768)],
        title="Test Document",
        url="https://example.com",
        content="This is a test document",
        tokens=5,
        date_indexed=datetime.datetime.now(),
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    # Return the dict without the embeddings
    doc = document.__dict__
    doc.pop("embeddings")
    return doc


# Routers
app.include_router(llm_router)
app.include_router(models_router)
