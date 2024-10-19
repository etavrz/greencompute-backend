from pydantic import BaseModel


class CarbonPredictionBody(BaseModel):
    memory: float = 10
    cpu: float = 10


class CarbonPredictionResponse(BaseModel):
    carbon: float
