import os

from dotenv import load_dotenv

load_dotenv()

ENVIRON = os.getenv("ENVIRON", "dev")
ROOT_PATH = "/api" if ENVIRON == "prod" else ""
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
