from typing import Literal

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    features: list[float]


class PredictionResponse(BaseModel):
    id: str
    defect: bool | None = None
    score: float | None = None
    model_version: str
    status: Literal["pending", "done", "failed"]
