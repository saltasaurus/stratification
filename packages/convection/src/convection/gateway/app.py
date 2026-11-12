import queue
import threading
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from .schemas import PredictFn, PredictionRequest, PredictionResponse
from .store import InMemoryStore
from .worker import run_worker


def create_app(predict: PredictFn, model_version: str, n_features: int, threshold: float = 0.5, store: InMemoryStore | None = None) -> FastAPI:
    app = FastAPI(title="convection gateway")
    if store is None:
        store = InMemoryStore()

    q = queue.Queue()
    threading.Thread(
        target=run_worker,
        args=(q, store, predict, threshold, model_version), 
        daemon=True,
        ).start()

    @app.post("/predictions", status_code=202, response_model=PredictionResponse)
    def create_prediction(req: PredictionRequest) -> PredictionResponse:
        if len(req.features) != n_features:
            raise HTTPException(status_code=422, detail=f"Expected {n_features} features; got {len(req.features)}")

        pr = PredictionResponse(
            id=str(uuid4()),
            model_version=model_version,
            status="pending",
        )

        store.save(pr)

        q.put((pr.id, req.features))

        return pr

    @app.get("/predictions/{id}", status_code=200, response_model=PredictionResponse)
    def get_prediction(id: str) -> PredictionResponse:
        pred = store.get(id)
        if pred is None:
            raise HTTPException(404, f"ID {id} missing from predictions")
        return pred
    
    return app