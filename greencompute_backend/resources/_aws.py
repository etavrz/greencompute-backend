import os

import boto3
from loguru import logger


def get_s3_client():
    try:
        client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name="us-east-1",  # e.g., "us-east-1"
        )
        return client
    except Exception as e:
        logger.error(f"Could not connect to S3: {e}")


def get_bedrock_client():
    try:
        client = boto3.client("bedrock-runtime")
        return client
    except Exception as e:
        logger.error(f"Could not connect to bedrock runtime: {e}")
