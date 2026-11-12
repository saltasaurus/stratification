from collections.abc import Callable

from fastapi import FastAPI

from .schemas import PredictionRequest, PredictionResponse

PredictFn = Callable[[list[float]], float]


def create_app(predict: PredictFn, model_version: str, n_features: int, threshold: float = 0.5) -> FastAPI:
    app = FastAPI(title="outcrop gateway")

    @app.post("/predictions", status_code=201, response_model=PredictionResponse)
    def create_prediction(req: PredictionRequest) -> PredictionResponse:
        # TODO (you):
        # 1. if len(req.features) != n_features -> raise HTTPException(422, ...)
        # 2. score = predict(req.features)
        # 3. return PredictionResponse with id=str(uuid4()), defect=score >= threshold, ...
        ...

    return app