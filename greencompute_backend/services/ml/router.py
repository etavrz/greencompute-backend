import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from loguru import logger

from greencompute_backend.config import AWS_S3_BUCKET, CARBON_MODEL, DATA_FILE
from greencompute_backend.resources._aws import get_s3_client

from .models import CarbonPredictionBody, CarbonPredictionResponse
from .svc import DataService, PredictionService

router = APIRouter(prefix="/ml", tags=["ml"])

try:
    emissions_model = PredictionService(CARBON_MODEL, get_s3_client())
except Exception as e:
    logger.error(f"Could not load model: {e}")


@router.post("/carbon-emissions", response_model=CarbonPredictionResponse)
async def root(payload: CarbonPredictionBody):
    prediction = emissions_model.predict(np.array([[payload.memory, payload.cpu]]))[0]
    return {"carbon": float(prediction)}


@router.get("/emissions-data")
async def emission_data(client=Depends(get_s3_client)):
    data_svc = DataService(client)
    data = data_svc.get_data(f"data/{DATA_FILE}")
    return data


@router.post("/upload")
async def upload_model(file: UploadFile, client=Depends(get_s3_client)):
    try:
        _ = file.file.read()
        file.file.seek(0)
        client.upload_fileobj(file.file, AWS_S3_BUCKET, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not upload file: {e}")

    return {"filename": file.filename}
