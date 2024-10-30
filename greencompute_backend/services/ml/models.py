from typing import Any

from pydantic import BaseModel


class CarbonPredictionBody(BaseModel):
    memory: float = 10
    cpu: float = 10


class ITElectricityBody(BaseModel):
    memory: float = 10
    cores: float = 10
    chips: float = 10


class PUEBody(BaseModel):
    state: str = "New York"
    cooler: str = "Air"
    economizer: str = "No"


class PredictionResponse(BaseModel):
    prediction: float
    features: dict[str, Any]
