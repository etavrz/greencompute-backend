from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, UploadFile

from greencompute_backend.config import AWS_S3_BUCKET, DATA_FILE, MODELS
from greencompute_backend.resources._aws import get_s3_client

from .models import CarbonPredictionBody, ITElectricityBody, PredictionResponse, PUEBody
from .svc import DataService, PredictionService

models = {}


@asynccontextmanager
async def lifespan_models(app: FastAPI):
    # Load the models and keep them in memory
    for model in MODELS:
        models[model] = PredictionService(MODELS[model], get_s3_client())
    yield
    # Clean up the embeddings model and release the resources
    models.clear()


router = APIRouter(prefix="/ml", tags=["ml"], lifespan=lifespan_models)


@router.post("/carbon-emissions", response_model=PredictionResponse)
async def root(payload: CarbonPredictionBody):
    prediction = models["carbon-emissions"].predict(np.array([[payload.memory, payload.cpu]]))[0]
    return {"prediction": float(prediction), "features": payload.model_dump()}


@router.post("/it-electricity", response_model=PredictionResponse)
async def it_electricity(payload: ITElectricityBody):
    prediction = models["it-electricity"].predict(np.array([[payload.memory, payload.cores, payload.cpu]]))[0]
    return {"prediction": prediction, "features": payload.model_dump()}


@router.post("/active-idle", response_model=PredictionResponse)
async def active_idle(payload: ITElectricityBody):
    prediction = models["active-idle"].predict(np.array([[payload.memory, payload.cores, payload.cpu]]))[0]
    return {"prediction": prediction, "features": payload.model_dump()}


@router.post("/pue", response_model=PredictionResponse)
async def pue(payload: PUEBody):
    # Format the data and make the prediction
    df = pd.DataFrame({"Cooling System": [payload.cooling_system], "state_name": [payload.state_name]})
    prediction = models["pue"].predict(df)[0]
    return {"prediction": prediction, "features": payload.model_dump()}


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
