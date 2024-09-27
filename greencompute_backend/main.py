import os

import boto3
from fastapi import FastAPI
from loguru import logger

app = FastAPI()

try:
    client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name="us-east-1",  # e.g., "us-east-1"
    )
except Exception as e:
    logger.error(f"Could not connect to S3: {e}")


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
