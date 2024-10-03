import datetime
import os

import boto3
from fastapi import Depends, FastAPI
from loguru import logger
from sqlalchemy.orm import Session

from .db.engine import SessionLocal, engine
from .db.tables import Base, Document

app = FastAPI()

Base.metadata.create_all(bind=engine)

try:
    client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name="us-east-1",  # e.g., "us-east-1"
    )
except Exception as e:
    logger.error(f"Could not connect to S3: {e}")


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/buckets")
async def list_buckets():
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
