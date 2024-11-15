from typing import Any

from pydantic import BaseModel, Field


class CarbonPredictionBody(BaseModel):
    memory: float = 10
    cpu: float = 10


class ITElectricityBody(BaseModel):
    memory: float = 10
    cores: float = 10
    cpu: float = 10


class PUEBody(BaseModel):
    state_name: str = "New York"
    cooling_system: str = Field("Air-cooled chiller", alias="Cooling System")


class PredictionResponse(BaseModel):
    prediction: float
    features: dict[str, Any]
