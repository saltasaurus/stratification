from collections.abc import Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from .schemas import PredictionRequest, PredictionResponse

PredictFn = Callable[[list[float]], float]


def create_app(predict: PredictFn, model_version: str, n_features: int, threshold: float = 0.5) -> FastAPI:
    app = FastAPI(title="convection gateway")

    @app.post("/predictions", status_code=201, response_model=PredictionResponse)
    def create_prediction(req: PredictionRequest) -> PredictionResponse:
        # 1. if len(req.features) != n_features -> raise HTTPException(422, ...)
        if len(req.features) != n_features:
            raise HTTPException(status_code=422, detail=f"Expected {n_features} features; got {len(req.features)}")
        # 2. score = predict(req.features)
        score = predict(req.features)
        # 3. return PredictionResponse with id=str(uuid4()), defect=score >= threshold, ...
        pr = PredictionResponse(
            id=str(uuid4()),
            defect=score >= threshold,
            score=score,
            model_version=model_version,
        )

        return pr

    return app