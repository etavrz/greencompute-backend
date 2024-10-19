from contextlib import asynccontextmanager

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, HTTPException, UploadFile

from greencompute_backend.config import AWS_S3_BUCKET, CARBON_MODEL, DATA_FILE
from greencompute_backend.resources._aws import get_s3_client

from .models import CarbonPredictionBody, CarbonPredictionResponse
from .svc import DataService, PredictionService

models = {}


@asynccontextmanager
async def lifespan_models(app: FastAPI):
    # Load the embeddings model
    models["carbon-emissions"] = PredictionService(CARBON_MODEL, get_s3_client())
    yield
    # Clean up the embeddings model and release the resources
    models.clear()


router = APIRouter(prefix="/ml", tags=["ml"], lifespan=lifespan_models)


@router.post("/carbon-emissions", response_model=CarbonPredictionResponse)
async def root(payload: CarbonPredictionBody):
    prediction = models["carbon-emissions"].predict(np.array([[payload.memory, payload.cpu]]))[0]
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
