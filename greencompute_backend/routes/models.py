from fastapi import APIRouter, Depends
from pydantic import BaseModel

from greencompute_backend.config import AWS_S3_BUCKET
from greencompute_backend.resources._aws import get_s3_client

router = APIRouter(prefix="/models", tags=["models"])


class CarbonPredictionBody(BaseModel):
    memory: float
    cpu: float


class CarbonPredictionResponse(BaseModel):
    carbon: float


@router.post("/carbon-footprint", response_model=CarbonPredictionResponse)
async def root(payload: CarbonPredictionBody):
    return {"carbon": payload.memory + payload.cpu}


@router.get("/dummy-model")
async def dummy_model(client=Depends(get_s3_client)):
    response = client.get_object(Bucket=AWS_S3_BUCKET, Key="dummy.txt")
    bytes_response = response["Body"].read()
    # return the string content of the file
    return {"content": bytes_response.decode("utf-8")}
