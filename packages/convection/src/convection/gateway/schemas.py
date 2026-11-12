from pydantic import BaseModel


class PredictionRequest(BaseModel):
    features: list[float]


class PredictionResponse(BaseModel):
    id: str
    defect: bool
    score: float
    model_version: str