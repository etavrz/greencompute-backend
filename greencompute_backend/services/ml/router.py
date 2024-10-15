import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from greencompute_backend.config import AWS_S3_BUCKET, CARBON_MODEL
from greencompute_backend.resources._aws import get_s3_client

from .models import CarbonPredictionBody, CarbonPredictionResponse
from .svc import PredictionService

router = APIRouter(prefix="/ml", tags=["ml"])

emissions_model = PredictionService(CARBON_MODEL, get_s3_client())


@router.post("/carbon-emissions", response_model=CarbonPredictionResponse)
async def root(payload: CarbonPredictionBody):
    prediction = emissions_model.predict(np.array([[payload.memory, payload.cpu]]))[0]
    return {"carbon": float(prediction)}


@router.get("/dummy-model")
async def dummy_model(client=Depends(get_s3_client)):
    response = client.get_object(Bucket=AWS_S3_BUCKET, Key="dummy.txt")
    bytes_response = response["Body"].read()
    return {"content": bytes_response.decode("utf-8")}


@router.post("/upload")
async def upload_model(file: UploadFile, client=Depends(get_s3_client)):
    try:
        _ = file.file.read()
        file.file.seek(0)
        client.upload_fileobj(file.file, AWS_S3_BUCKET, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not upload file: {e}")

    return {"filename": file.filename}
