from pydantic import BaseModel


class CarbonPredictionBody(BaseModel):
    memory: float
    cpu: float


class CarbonPredictionResponse(BaseModel):
    carbon: float
